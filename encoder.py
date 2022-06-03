import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import build_tree
import random

PI = torch.acos(torch.tensor(-1.0))
inf = 1e10

def swap_dim(ans, dim1, dim2):
    return ans.transpose(dim1, dim2)

def transpose(ans, dim=-2):
    return swap_dim(ans, -1, dim)

def make_lowrk(idim, odim, rank):
    if rank * (idim + odim) < idim * odim:
        return nn.Sequential(
            torch.nn.Linear(idim, rank, bias=False),
            torch.nn.Linear(rank, odim)
        )
    return nn.Linear(idim, odim)

def convert_lowrk(model, rank):
    return make_lowrk(model.in_features, model.out_features, rank).to(model.weight.device)

class FC(torch.nn.Module):
    def __init__(self, idim, odim, init=0.25, dim=-1, bn=True, flatten=None):
        super(FC, self).__init__()
        self.dim = dim
        self.flatten = flatten
        self.idim = idim
        self.odim = odim

        self.linear = nn.Linear(idim, odim)
        if init is None:
            self.relu = lambda x : x
        elif init is False:
            self.relu = nn.ReLU()
        else:
            self.relu = nn.PReLU(init=init)
        if bn:
            self.bn = torch.nn.BatchNorm1d(odim)
            def calc_bn(x):
                if len(x.shape) == 2:
                    return self.bn(x)
                return self.bn(x.transpose(-1, -2)).transpose(-1, -2)

            self._bn = calc_bn
        else:
            self._bn = lambda x : x

    def forward(self, ans, dim=None):
        if dim is None:
            dim = self.dim
        ans = transpose(ans, dim=dim)

        if self.flatten is not None:
            shape = ans.shape
            fl = len(self.flatten)
            ans = ans.reshape(*shape[:-fl], self.idim)

        # assert ans.size(-1) == self.idim

        ans = self.relu(self._bn(self.linear(ans)))

        if self.flatten is not None:
            ans = ans.reshape(*shape[:-fl], *self.flatten)

        ans = transpose(ans, dim=dim)
        return ans

class MLP(torch.nn.Module):
    def __init__(self, dims, init=0.25, last_relu=False, bn=True, last_bn=True, last_dropout=None):
        super(MLP, self).__init__()

        layers = []

        for i in range(1, len(dims)):
            is_last = (i == len(dims) - 1)
            no_relu = is_last and not last_relu
            no_bn = is_last and not last_bn
            if i == len(dims) - 1 and last_dropout is not None:
                layers.append(torch.nn.Dropout(last_dropout))
            layers.append(FC(dims[i - 1], dims[i], init=None if no_relu else init, bn=bn and not no_bn))

        self.layers = nn.ModuleList(layers)

    def forward(self, ans):
        for l in self.layers:
            ans = l(ans)
        return ans

class Alignment(torch.nn.Module):
    def __init__(self, k, k_in=None, use_attn=False):
        super(Alignment, self).__init__()
        self.k = k
        if k_in is None:
            k_in = k
        self.k_in = k_in

        if use_attn:
            self.attn = Attention(512, 512, 512, 64, embed_dim=64, head=8)
        else:
            self.attn = None

        self.conv1 = torch.nn.Conv1d(k_in, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 512 if use_attn else 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512 if use_attn else 1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, points, points_in=None):
        k = self.k
        # print(points.shape, points_in.shape)
        if points_in is None:
            points_in = points
        assert len(points.shape) == 3
        assert points_in.shape[-1] == self.k_in
        x = points_in.transpose(-1, -2)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.attn is not None:
            xt = x.transpose(-1, -2)
            attn = self.attn(xt, xt, xt)
            x = torch.cat([x, attn.transpose(-1, -2)], dim=1)

        x = torch.max(x, 2)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        I = torch.eye(k, device='cuda')
        x = x.reshape(-1, k, k) + I

        self.align_reg_loss = 0. #(x.bmm(x.transpose(-1, -2)) - I).pow(2).sum(dim=(-1, -2)).mean()

        return torch.bmm(points, x)


class EncoderLayer(torch.nn.Module):

    def __init__(self, ind, layer_type, idim, odim, sdim=None, 
            dropout=0.5, relu_weight=0.005, srate=0.25, dense_mlp=False, skip_same_dim=False, 
            layer0_mlp_dim=1024, extra_dim=0, catmlp=False):
        super(EncoderLayer, self).__init__()

        self.ind = ind
        self.layer_type = layer_type
        self.idim = idim
        self.odim = odim
        self.sdim = sdim
        self.srate = srate
        self.feature_dim = odim
        self.relu_weight = relu_weight
        self.catmlp = catmlp
        if layer_type == 'leaf':
            if idim == 3:
                self.extra_dim = extra_dim
                if extra_dim > 0:
                    idim += extra_dim
                    self.pts_align = Alignment(3, 3 + extra_dim)
                else:
                    self.pts_align = Alignment(3)
            else:
                self.extra_dim = 0
                self.pts_align = torch.nn.Identity()
            
            self.mlp = MLP([idim, layer0_mlp_dim, odim], init=relu_weight)
            # self.mlp = MLP([idim, odim * 4, odim], init=relu_weight)
        else:
            if idim == odim and skip_same_dim and layer_type != 'sampled':
                self.upload = torch.nn.Identity()
            else:
                if not self.catmlp:
                    self.upload = MLP([idim, odim * 2, odim] if dense_mlp else [idim, odim], init=relu_weight)
                else:
                    self.upload = nn.ModuleList([
                        nn.ModuleList([
                            MLP([idim, odim], init=relu_weight) for _ in range(2)
                        ]) for _ in range(3)
                    ])

            if layer_type == 'sampled':
                # self.upload_sample = MLP([sdim, idim, odim], init=relu_weight)
                self.merge_sample = MLP([odim + sdim, odim * 2, odim] if dense_mlp else [odim + sdim, odim], init=relu_weight)

            # if ind in [2, 4, 6]:
            #     self.align = Alignment(odim)
            # else:
            #     self.align = lambda x : x            
            

        self.dropout = lambda x : x # nn.Dropout(dropout)

    def forward(self, ans, line=None, sample=None, vec=None, dmap=None, drev=None):
        if self.layer_type[0] == 'l':
            if self.extra_dim > 0:
                _1, _2 = ans.split([3, self.extra_dim], -1)
                ans = torch.cat([self.pts_align(_1, points_in=ans), _2], dim=-1)
            else:
                ans = self.pts_align(ans)
            ans = self.mlp(ans)
        else:
            # ans = self.merge(torch.cat([ans[:, self.child_l], ans[:, self.child_r]], dim=-1))

            if not self.catmlp:
                ans = self.upload(ans[:, self.child_lr]).reshape(ans.size(0), 2, -1, self.odim).max(dim=1)[0]
            else:
                out = torch.zeros([ans.size(0), ans.size(1) // 2, self.odim], device='cuda')

                lch = ans[:, self.child_l]
                rch = ans[:, self.child_r]
                for k, ch in enumerate([lch, rch]):
                    for d in range(3):
                        sgn = (line[:, :, d] > 0).long()
                        for s in range(2):
                            mask = (sgn == (s ^ k))
                            out[mask] += self.upload[d][s](ch)[mask]
                ans = out

            if self.layer_type[0] == 's':
                # sr = self.srate / (1 + self.srate)
                # smp = self.upload_sample(sample[:, self.child_s])
                # replaced = torch.bernoulli(torch.full_like(ans, sr).cuda()).bool()
                # ans = ans.masked_scatter(replaced, smp[replaced])

                smp = sample[:, self.child_s]
                ans = self.merge_sample(torch.cat([ans, smp], dim=-1))
                # ans = self.max(ans, self.upload_sample(smp))

            # ans = self.align(ans)

        return self.dropout(ans)


class Encoder(torch.nn.Module):

    def __init__(self, N, sample_layers, dim, OUTPUT, 
        extra_dim=0, point_dim=3, dim_layer0=16, layer0_mlp_dim=1024, dim_repeat_cut=4, rotate=True, use_symmetry_loss=False, 
        channel=1, sample_child_first=True, skip_same_dim=False, catmlp=False):
        super(Encoder, self).__init__()

        assert channel == 1

        self.N = N
        self.OUTPUT = OUTPUT
        self.num_layers = -1
        self.layers = None
        self.dim = dim
        self.odim = dim
        self.idim = point_dim

        layer_dict = None

        self.tree = build_tree.BuildTree(N, sample_layers, sample_child_first=sample_child_first, 
            use_symmetry_loss=use_symmetry_loss, record_vec=catmlp)

        # generate tree structure
        # os.system("g++ build_tree.cpp -o build_tree -O2 -std=c++14 -Wall")
        # os.system(f"./build_tree {sample_layers} struct {N} > {OUTPUT}/struct.txt")
        cur_layer = -1

        def pmap(layer, p):
            p = int(p)
            if layer < 0 or p == -1:
                return [-1]
            
            if p not in layer_dict[layer]:
                layer_dict[layer][p] = [len(layer_dict[layer])]
            return layer_dict[layer][p]

        for line in self.tree.structure():
            if len(line) == 0:
                continue
            if line[0] == 'size':
                self.N = N = int(line[1])
            elif line[0] == 'sample':
                self.sample_layers = int(line[1])
                assert self.sample_layers == sample_layers
            elif line[0] == 'layer':
                if self.num_layers == -1:
                    self.num_layers = int(line[1]) + 1
                    layer_dict = [dict() for _ in range(self.num_layers)]
                cur_layer = int(line[1])
            else:
                data = pmap(cur_layer, line[1])
                data.append(int(line[2])) # real point id
                data.append(pmap(cur_layer + 1, line[3])[0]) # child l
                data.append(pmap(cur_layer + 1, line[4])[0]) # child r
                data.append(pmap(cur_layer + self.sample_layers, line[5])[0]) # child sample


        logging.info(f'self.N = {N}')
        logging.info(f'self.num_layers = {self.num_layers}')
        logging.info(f'self.sample_layers = {self.sample_layers}')

        self.layers = []

        feature_dim = dim_layer0
        dim_repeat = 1
        odims = [-1] * 100

        for i in reversed(range(self.num_layers)):
            layer_type = 'sampled'
            if i + self.sample_layers >= self.num_layers:
                layer_type = 'unsampled'
            if i + 1 == self.num_layers:
                layer_type = 'leaf'


            if layer_type != 'leaf':
                idim = feature_dim
                if odims[-dim_repeat] in ([-1, feature_dim] if dim_repeat_cut != 0 else [feature_dim]):
                    feature_dim = min(feature_dim * 2, self.dim)
                odim = feature_dim
            else:
                idim = point_dim
                odim = feature_dim

            sdim = None
            if layer_type == 'sampled':
                sdim = odims[-self.sample_layers]

            odims.append(odim)
            data = layer_dict[i]

            layer = EncoderLayer(len(self.layers), layer_type, idim, odim, 
                sdim=sdim, 
                relu_weight=0.25,
                dropout=min(0.1 * (1.2 ** len(self.layers)), 0.5),
                srate=2 ** -sample_layers,
                dense_mlp=False, # len(self.layers) <= dim_repeat_cut,
                skip_same_dim=skip_same_dim,
                layer0_mlp_dim=layer0_mlp_dim,
                extra_dim=extra_dim,
                catmlp=catmlp,
            )
            layer.num_nodes = len(data)
            data = sorted(data.values(), key=lambda x : x[0])

            if layer_type != 'leaf':
                layer.child_l = torch.tensor([l for _, _, l, _, _ in data]).cuda()
                layer.child_r = torch.tensor([r for _, _, _, r, _ in data]).cuda()
                layer.child_lr = torch.cat([layer.child_l, layer.child_r], dim=0)

                if layer_type == 'sampled':
                    layer.child_s = torch.tensor([s for _, _, _, _, s in data]).cuda()

            self.layers.append(layer)

            if len(self.layers) >= dim_repeat_cut:
                dim_repeat = 2

            logging.info(f"layer {layer.ind} ({layer_type}) # = {layer.num_nodes} odim = {odim}")

        self.layers = nn.ModuleList(self.layers)

    def directions(self):
        d = build_tree.get_directions().clone()
        return d

    def forward(self, ans, inputs, extra, perm=None):
        self.layer_output = []        
        self.arrange = inputs[0]
        ans = ans.cuda()

        if self.idim != 3:
            perm = None

        if perm is not None:
            axisperm, axissgn, dmap, drev = build_tree.transforms[perm]
            ans[:, :, :3] = ans[:, :, axisperm] * axissgn
        else:
            axisperm = axissgn = dmap = drev = None

        for i, (line, layer) in enumerate(zip(inputs, self.layers)):
            # print(f"forward #{i} ans = {ans.shape}")

            vec = None
            if layer.layer_type[0] != 'l':
                vec = line.cuda()
                
            sample = None
            if layer.layer_type[0] == 's':
                sample = self.layer_output[-self.sample_layers]

            ans = layer(ans, line=line.cuda(), sample=sample, vec=vec, dmap=dmap, drev=drev)

            if layer.layer_type[0] == 'l':
                # Apply gather after layer, to increase efficiency
                arrange = line.cuda().unsqueeze(-1).expand(line.shape + (ans.shape[-1], ))
                ans = ans.gather(1, arrange)

            self.layer_output.append(ans)

            # assert ans.isnan().sum() == 0

       
        try:
            self.align_reg_loss = self.layers[0].pts_align.align_reg_loss
        except:
            self.align_reg_loss = torch.tensor(0., device='cuda')
        return ans.squeeze(1)

class AlignWithEncoder(torch.nn.Module):
    def __init__(self, k, encoder):
        super().__init__()
        self.k = k
        self.encoder = encoder
        self.feed = MLP([encoder.dim, encoder.dim // 2, encoder.dim // 4, k * k], last_bn=False)
        self.dim = self.encoder.dim
        self.upload = torch.nn.Identity()

        if isinstance(self.encoder, EncoderKdtAlign):
            self.encoder.align.upload = torch.nn.Linear(self.encoder.align.dim, self.dim)

    
    def forward(self, pts, *args, **kwargs):
        k = self.k
        pts = pts.cuda()
        x = self.encoder(pts, *args, **kwargs).reshape(-1, self.dim)
        
        if isinstance(self.encoder, EncoderKdtAlign):
            self.features = self.upload(x + self.encoder.align.features)
        else:
            self.features = self.upload(x)


        x = self.feed(x)
        x = x.reshape(-1, k, k) + torch.eye(k, device='cuda')
        pts = torch.bmm(pts, x)
        self.pts = pts
        return pts

class EncoderKdtAlign(torch.nn.Module):
    def __init__(self, *args, num_layers=1, **kwargs):
        super().__init__()
        assert num_layers >= 1

        if True:
            from copy import copy
            args1 = list(copy(args))
            args1[2] >>= 1 # dim /= 2
            kwargs1 = copy(kwargs)
            kwargs1['dim_layer0'] >>= 1 # dim_layer0 /= 2

            if num_layers == 1:
                align_encoder = Encoder(*args1, **kwargs1)
                # align_encoder.layers[0].pts_align = torch.nn.Identity()
            else:
                align_encoder = EncoderKdtAlign(*args1, num_layers=num_layers-1, **kwargs1)


        self.align = AlignWithEncoder(3, align_encoder)
        self.encoder = Encoder(*args, **kwargs)
        self.dim = self.encoder.dim
        self.tree = self.encoder.tree
        self.encoder.layers[0].pts_align = torch.nn.Identity()
        self.align_dim = self.align.encoder.dim

    
    def forward(self, pts, *args, **kwargs):
        pts = self.align(pts, *args, **kwargs)
        self.align_feature = self.align.features
        return self.encoder(pts, *args, **kwargs)

class EncoderRec(torch.nn.Module):
    def __init__(self, *args, num_layers=2, **kwargs):
        super().__init__()
        
        self.num_layers = num_layers

        encoder = Encoder(*args, **kwargs)
        self.first_align = encoder.layers[0].pts_align
        self.encoder = encoder
        self.align = AlignWithEncoder(3, encoder)

        
        self.encoder.layers[0].pts_align = torch.nn.Identity()
        self.tree = self.encoder.tree
        self.dim = self.encoder.dim

    def forward(self, pts, *args, **kwargs):
        pts = pts.cuda()
        pts = self.first_align(pts)
        for _ in range(self.num_layers):
            pts = self.align(pts, *args, **kwargs)
        return self.encoder(pts, *args, **kwargs)





                

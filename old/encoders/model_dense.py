import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import build_tree
import random

PI = torch.acos(torch.tensor(-1.0))
inf = 1e10

global logging_init_flag
logging_init_flag = False

def init_logging(OUTPUT):
    global logging_init_flag
    if logging_init_flag:
        return
    logging_init_flag = True

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s:\t%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(f"{OUTPUT}/training.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

def swap_dim(ans, dim1, dim2):
    return ans.transpose(dim1, dim2)

def transpose(ans, dim=-2):
    return swap_dim(ans, -1, dim)

class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = alpha
        
    def forward(self, x):
        if not self.training:
            return x

        epsilon = torch.randn(x.size()).cuda() * self.alpha + 1
        return x * epsilon

class FC(torch.nn.Module):
    def __init__(self, idim, odim, init=0.25, dim=-1, bn=True, flatten=None, dropout=0):
        super(FC, self).__init__()
        self.dim = dim
        self.flatten = flatten
        self.idim = idim
        self.odim = odim
        self.linear = nn.Linear(idim, odim)
        if init is None:
            self.activate = lambda x : x
        else:
            self.activate = nn.PReLU(init=init)
        if bn:
            self.batch_norm = torch.nn.BatchNorm1d(odim)
            def calc_bn(x):
                if len(x.shape) == 2:
                    return self.batch_norm(x)
                return self.batch_norm(x.transpose(-1, -2)).transpose(-1, -2)

            self.bn = calc_bn
        else:
            self.bn = lambda x : x

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = lambda x : x

    def forward(self, ans, dim=None):
        if dim is None:
            dim = self.dim
        ans = transpose(ans, dim=dim)

        if self.flatten is not None:
            shape = ans.shape
            fl = len(self.flatten)
            ans = ans.reshape(*shape[:-fl], self.idim)

        # assert ans.size(-1) == self.idim

        ans = self.activate(self.dropout(self.bn(self.linear(ans))))

        if self.flatten is not None:
            ans = ans.reshape(*shape[:-fl], *self.flatten)

        ans = transpose(ans, dim=dim)
        return ans


class MLP(torch.nn.Module):
    def __init__(self, dims, init=0.25, last_activate=False, last_bn=True, dropout=0):
        super(MLP, self).__init__()

        layers = []

        for i in range(1, len(dims)):
            last = (i == len(dims) - 1)
            last2 = (i == len(dims) - 2)
            use_activate = last_activate or (not last)
            use_bn = last_bn or (not last)
            layers.append(FC(dims[i - 1], dims[i], 
                init=init if use_activate else None, 
                bn=use_bn,
                dropout=dropout if last2 else 0))

        self.layers = nn.ModuleList(layers)

    def forward(self, ans):
        for l in self.layers:
            ans = l(ans)
        return ans

class Alignment(torch.nn.Module):
    def __init__(self, k, use_attn=False):
        super(Alignment, self).__init__()
        self.k = k
        if use_attn:
            self.attn = Attention(512, 512, 512, 64, embed_dim=64, head=8)
        else:
            self.attn = None
        self.conv1 = torch.nn.Conv1d(k, 128, 1)
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

    def forward(self, points):
        k = self.k
        assert len(points.shape) == 3
        assert points.shape[-1] == k
        x = points.transpose(-1, -2)

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

        x = x.reshape(-1, k, k) + torch.eye(k).cuda()

        return torch.bmm(points, x)


class EncoderLayer(torch.nn.Module):

    def __init__(self, ind, layer_type, idim, odim, sdim=None, dropout=0.5, relu_weight=0.005, sample_rate=0.25):
        super(EncoderLayer, self).__init__()

        self.ind = ind
        self.layer_type = layer_type
        self.idim = idim
        self.odim = odim
        self.sample_rate = sample_rate
        if layer_type == 'leaf':
            assert idim == 3
            self.point_to_embed = MLP([3, odim, odim], init=relu_weight)
        else:
            ndir = build_tree.num_directions()
            
            self.upload = MLP([idim, odim, odim], init=relu_weight)

            if layer_type == 'sampled':
                self.merge = MLP([odim + sdim, odim * 2, odim], init=relu_weight)
            
            if ind in [2, 4, 6]:
                self.align = Alignment(odim)
            else:
                self.align = lambda x : x            
            

        self.dropout = lambda x : x # GaussianDropout(dropout)
 

    def forward(self, last, sample=None, vec=None, dmap=None, drev=None):
        if self.layer_type[0] == 'l':
            ans = self.point_to_embed(last)
            ans = self.dropout(ans)
            return ans, ans

        def zero_with_grad(x):
            if not self.training:
                return torch.zeros_like(x).to(x.device)
            return x - x.detach()

        def softhardmax(inputs, fact=5):
            ans = torch.stack(inputs, dim=-1)
            val = ans.detach().max(dim=-1)[0]
            ans = ans * (ans.detach() * fact).softmax(dim=-1)
            ans = ans.sum(dim=-1)
            return val.detach() + zero_with_grad(ans)

        lch = self.upload(last[:, self.child_l])
        rch = self.upload(last[:, self.child_r])
        ans = softhardmax([lch, rch])

        if self.layer_type[0] != 's':
            ans = self.align(ans)
            ans = self.dropout(ans)
            return ans, ans
        
        sample = sample[:, self.child_s]
        smp = lambda : torch.cat([ans, sample], dim=-1)
        ans = self.merge(smp())
        ans = self.align(ans)
        ans = self.dropout(ans)
        return ans, smp()


class Encoder(torch.nn.Module):

    def __init__(self, N, sample_layers, dim, OUTPUT, rotate=True, channel=1):
        super(Encoder, self).__init__()

        assert channel == 1

        self.OUTPUT = OUTPUT
        self.num_layers = -1
        self.layers = None
        self.dim = dim

        layer_dict = None

        self.tree = build_tree.BuildTree(N, sample_layers)

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

        feature_dim = 8
        dim_repeat = 2
        odims = [-1] * 100
        sdims = [-1] * 100

        self.align_pts = Alignment(3)

        for i in reversed(range(self.num_layers)):
            layer_type = 'sampled'
            if i + self.sample_layers >= self.num_layers:
                layer_type = 'unsampled'
            if i + 1 == self.num_layers:
                layer_type = 'leaf'


            if layer_type != 'leaf':
                idim = feature_dim
                if odims[-dim_repeat] in [-1, feature_dim]:
                    feature_dim = min(feature_dim * 2, self.dim)
                odim = feature_dim
            else:
                idim = 3
                odim = feature_dim

            if odim > 64:
                dim_repeat = 1

            sdim = 0
            if layer_type == 'sampled':
                sdim = sdims[-self.sample_layers]

            sdims.append(sdim + odim)

            odims.append(odim)
            data = layer_dict[i]

            layer = EncoderLayer(len(self.layers), layer_type, idim, odim, 
                sdim=sdim, 
                relu_weight=0.25,
                dropout=min(0.1 * (1.2 ** i), 0.5),
                sample_rate=2 ** -sample_layers,
            )
            layer.num_nodes = len(data)
            data = sorted(data.values(), key=lambda x : x[0])

            if layer_type != 'leaf':
                layer.child_l = torch.tensor([l for _, _, l, _, _ in data]).cuda()
                layer.child_r = torch.tensor([r for _, _, _, r, _ in data]).cuda()

                if layer_type == 'sampled':
                    layer.child_s = torch.tensor([s for _, _, _, _, s in data]).cuda()

            self.layers.append(layer)

            logging.info(f"layer {i} ({layer_type}) # = {layer.num_nodes} odim = {odim} sdim = {sdim}")

        self.layers = nn.ModuleList(self.layers)

    def directions(self):
        d = build_tree.get_directions().clone()
        return d

    def forward(self, ans, inputs, perm=None):

        samples = []        
        ans = ans.cuda()

        if perm is not None:
            axisperm, axissgn, dmap, drev = build_tree.transforms[perm]
            ans = ans[:, :, axisperm] * axissgn
        else:
            axisperm = axissgn = dmap = drev = None

        ans = self.align_pts(ans)

        for i, (line, layer) in enumerate(zip(inputs, self.layers)):

            vec = None
            if layer.layer_type[0] != 'l':
                vec = line.cuda()
                
            sample = None
            if layer.layer_type[0] == 's':
                sample = samples[-self.sample_layers]

            ans, sample = layer(ans, sample=sample, vec=vec, dmap=dmap, drev=drev)

            if layer.layer_type[0] == 'l':
                # Apply gather after layer, to increase efficiency
                arrange = line.cuda().unsqueeze(-1).expand(line.shape + (ans.shape[-1], ))
                sample = ans = ans.gather(1, arrange)

            samples.append(sample)

            # print(f"forward #{i} ans = {ans.shape} sample = {sample.shape}")
            assert ans.isnan().sum() == 0

        return ans.squeeze(1)


def debug_main():
    OUTPUT = './scratch'
    encoder = Encoder(2048, 2, 128, OUTPUT).cuda()
    for name, param in encoder.named_parameters():
        print(f"Parameter: {name} {param.size()}")

    
if __name__ == '__main__':
    debug_main()


                

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

class FC(torch.nn.Module):
    def __init__(self, idim, odim, init=0.25, dim=-1, bn=True, flatten=None, dropout=0):
        super(FC, self).__init__()
        self.dim = dim
        self.flatten = flatten
        self.idim = idim
        self.odim = odim
        self.linear = nn.Linear(idim, odim)
        if init is None:
            self.relu = lambda x : x
        else:
            self.relu = nn.PReLU(init=init)
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

        ans = self.relu(self.dropout(self.bn(self.linear(ans))))

        if self.flatten is not None:
            ans = ans.reshape(*shape[:-fl], *self.flatten)

        ans = transpose(ans, dim=dim)
        return ans


class MLP(torch.nn.Module):
    def __init__(self, dims, init=0.25, last_relu=False, last_bn=True, dropout=0):
        super(MLP, self).__init__()

        layers = []

        for i in range(1, len(dims)):
            last = (i == len(dims) - 1)
            last2 = (i == len(dims) - 2)
            use_relu = last_relu or (not last)
            use_bn = last_bn or (not last)
            layers.append(FC(dims[i - 1], dims[i], 
                init=init if use_relu else None, 
                bn=use_bn,
                dropout=dropout if last2 else 0))

        self.layers = nn.ModuleList(layers)

    def forward(self, ans):
        for l in self.layers:
            ans = l(ans)
        return ans

def attention(Q, K, V=None, softmax=True):
    idim = Q.size(-2)
    odim = K.size(-2)
    embed = Q.size(-1)

    assert K.size(-1) == embed
    
    ans = Q.matmul(transpose(K)) 
    ans /= embed ** 0.5

    if softmax:
        ans = ans.softmax(dim=-1)

    if V is None:
        return ans

    assert V.size(-2) == odim
    return ans.matmul(V)


class Attention(torch.nn.Module):
    def __init__(self, dimq, dimk, dimv, head_dim, head=1, embed_dim=None):
        super(Attention, self).__init__()

        if embed_dim is None:
            assert dimq == dimk
            embed_dim = dimk

        self.head = head
        self.head_dim = head_dim
        self.linearQ = MLP([dimq, embed_dim * head])
        self.linearK = MLP([dimk, embed_dim * head])
        self.linearV = MLP([dimv, head_dim * head])
        self.batch_norm = torch.nn.BatchNorm1d(head_dim * head)


    def forward(self, Q, K, V):
        def split_head(M):
            M = M.reshape(*M.shape[:-1], self.head, self.head_dim)
            M = swap_dim(M, -2, -3)
            return M

        def merge_head(M):
            M = swap_dim(M, -2, -3)
            M = M.reshape(*M.shape[:-2], self.head * self.head_dim)
            return M

        Q = split_head(self.linearQ(Q))
        K = split_head(self.linearK(K))
        V = split_head(self.linearV(V))
        ans = merge_head(attention(Q, K, V))
        ans = self.batch_norm(ans.transpose(-1, -2)).transpose(-1, -2)
        
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

        # tmp = self.fc3.state_dict()
        # tmp['bias'] += torch.eye(k).view(-1)
        # self.fc3.load_state_dict(tmp)

        with torch.no_grad():
            self.fc3.bias.set_(self.fc3.bias + torch.eye(k).view(-1))

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

        x = x.reshape(-1, k, k)
        
        # xd = x.detach()
        # xtx = xd.bmm(xd.transpose(-1, -2))
        # trace = (xtx * torch.eye(k).cuda()).view(-1, k * k).sum(-1)
        # fact = (trace / k).pow(0.5)
        # x = x / fact[:, None, None]

        self.matrix = x

        return torch.bmm(points, x)



class EncoderLayer(torch.nn.Module):

    def __init__(self, ind, layer_type, idim, odim, sdim=None, edim=0, dropout=0.5, relu_weight=0.005):
        super(EncoderLayer, self).__init__()

        self.ind = ind
        self.layer_type = layer_type
        self.idim = idim
        self.odim = odim
        self.align_matrix = None

        if layer_type == 'leaf':
            assert idim == 3
            pd = int(0.6 * odim + 0.5)
            ed = odim - pd
            self.upload_point = MLP([3, pd * 2, pd], init=relu_weight)
            self.upload_extra = MLP([edim, ed * 2, ed], init=relu_weight)
        else:
            ndir = build_tree.num_directions()
                
            self.upload = MLP([idim, idim * 2, idim * 2], init=relu_weight)
            self.save = MLP([idim * 2, odim], init=relu_weight)

            if layer_type == 'sampled':
                self.upload_sample = MLP([sdim, sdim * 2, idim * 2, odim], init=relu_weight)
                # self.merge = MLP([odim + sdim, odim * 2, odim])
            
            if ind in [2, 4, 6]:
                self.alignment = Alignment(odim)
                def align(x):
                    x = self.alignment(x)
                    self.align_matrix = self.alignment.matrix
                    return x
                self.align = align

            else:
                self.align = lambda x : x            
            
        self.distill_loss = torch.tensor(0.0).cuda()
        self.dropout = dropout


    def forward(self, last, sample=None, vec=None, dmap=None, drev=None, extra=None):
        if self.layer_type[0] == 'l':
            ans = self.upload_point(last)
            ext = self.upload_extra(extra)
            ans = torch.cat([ans, ext], dim=-1)
        else:
            lch = self.upload(last[:, self.child_l])
            rch = self.upload(last[:, self.child_r])
            ans = self.save(torch.maximum(lch, rch))

            if self.layer_type[0] == 's':
                smp = sample[:, self.child_s]
                # ans = self.merge(torch.cat([smp, ans], dim=-1))
                smp = self.upload_sample(smp)

                if self.training:
                    diff = self.dropout * (ans - smp.detach()) + (1 - self.dropout) * (ans.detach() - smp)
                else:
                    diff = ans - smp
                    
                self.distill_loss = diff.norm(dim=(-2, -1)).mean()

                rand = torch.rand_like(ans).cuda() * (1 + self.dropout)
                cho = rand <= 1
                ans = (ans * cho) + (smp * ~cho)

            ans = self.align(ans)

        return ans


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

        feature_dim = 16
        dim_repeat = 3
        odims = [-1] * 100

        self.align_pts = Alignment(3)

        for i in reversed(range(self.num_layers)):
            layer_type = 'sampled'
            if i + self.sample_layers >= self.num_layers:
                layer_type = 'unsampled'
            if i + 1 == self.num_layers:
                layer_type = 'leaf'


            if layer_type != 'leaf':
                idim = feature_dim
                if odims[-dim_repeat] == feature_dim:
                    feature_dim = min(feature_dim * 2, self.dim)
                odim = feature_dim
            else:
                idim = 3
                odim = feature_dim

            if odim > 64:
                dim_repeat = 1
            elif odim > 16:
                dim_repeat = 2
            

            sdim = None
            if layer_type == 'sampled':
                sdim = odims[-self.sample_layers]

            odims.append(odim)
            data = layer_dict[i]

            layer = EncoderLayer(len(self.layers), layer_type, idim, odim, 
                sdim=sdim, 
                edim=self.num_layers - 1,
                relu_weight=0.25,
                dropout=min(0.05 * (1.2 ** len(self.layers)), 0.2),
                # srate=2 ** -sample_layers,
            )
            layer.num_nodes = len(data)
            data = sorted(data.values(), key=lambda x : x[0])

            if layer_type != 'leaf':
                layer.child_l = torch.tensor([l for _, _, l, _, _ in data]).cuda()
                layer.child_r = torch.tensor([r for _, _, _, r, _ in data]).cuda()

                if layer_type == 'sampled':
                    layer.child_s = torch.tensor([s for _, _, _, _, s in data]).cuda()

            self.layers.append(layer)

            logging.info(f"layer {i} ({layer_type}) # = {layer.num_nodes} odim = {odim}")

        self.layers = nn.ModuleList(self.layers)

    def directions(self):
        d = build_tree.get_directions().clone()
        return d

    def forward(self, ans, inputs, extra, perm=None):

        self.distill_loss = torch.tensor(0.).cuda()

        layer_output = []        
        ans = ans.cuda()

        if perm is not None:
            axisperm, axissgn, dmap, drev = build_tree.transforms[perm]
            ans = ans[:, :, axisperm] * axissgn
        else:
            axisperm = axissgn = dmap = drev = None

        ans = self.align_pts(ans)
        extra = extra.cuda()

        for i, (line, layer) in enumerate(zip(inputs, self.layers)):

            vec = None
            if layer.layer_type[0] != 'l':
                vec = line.cuda()
                
            sample = None
            if layer.layer_type[0] == 's':
                sample = layer_output[-self.sample_layers]

            ans = layer(ans, sample=sample, vec=vec, dmap=dmap, drev=drev, extra=extra)

            if layer.layer_type[0] == 'l':
                # Apply gather after layer, to increase efficiency
                arrange = line.cuda().unsqueeze(-1).expand(line.shape + (ans.shape[-1], ))
                ans = ans.gather(1, arrange)

            self.distill_loss += layer.distill_loss
            layer_output.append(ans)

            # print(f"forward #{i} ans = {ans.shape} vec = {vec if vec is None else vec.shape}")
            # assert ans.isnan().sum() == 0

        return ans.squeeze(1)


def debug_main():
    OUTPUT = './scratch'
    encoder = Encoder(2048, 2, 128, OUTPUT).cuda()
    for name, param in encoder.named_parameters():
        print(f"Parameter: {name} {param.size()}")

    
if __name__ == '__main__':
    debug_main()




                

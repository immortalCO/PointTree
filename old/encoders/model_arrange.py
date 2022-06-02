import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import build_tree
import random

PI = torch.acos(torch.tensor(-1.0))

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
    if dim1 != dim2:
        perm = list(range(len(ans.shape)))
        perm[dim1], perm[dim2] = perm[dim2], perm[dim1]
        ans = ans.permute(*perm)
    return ans

def transpose(ans, dim=-2):
    return swap_dim(ans, -1, dim)

class FC(torch.nn.Module):
    def __init__(self, idim, odim, init=0.25, dim=-1, flatten=None):
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

    def forward(self, ans, dim=None):
        if dim is None:
            dim = self.dim
        ans = transpose(ans, dim=dim)

        if self.flatten is not None:
            shape = ans.shape
            fl = len(self.flatten)
            ans = ans.reshape(*shape[:-fl], self.idim)

        # assert ans.size(-1) == self.idim

        ans = self.activate(self.linear(ans))

        if self.flatten is not None:
            ans = ans.reshape(*shape[:-fl], *self.flatten)

        ans = transpose(ans, dim=dim)
        return ans

class MLP(torch.nn.Module):
    def __init__(self, dims, init=0.25, disable_last_activate=True):
        super(MLP, self).__init__()

        layers = []

        for i in range(1, len(dims)):
            disable = (i == len(dims) - 1) and disable_last_activate
            layers.append(FC(dims[i - 1], dims[i], init=None if disable else init))

        self.layers = nn.ModuleList(layers)

    def forward(self, ans):
        for l in self.layers:
            ans = l(ans)
        return ans

# def attention(Q, K, V=None, softmax=True):
#     idim = Q.size(-2)
#     odim = K.size(-2)
#     embed = Q.size(-1)

#     assert K.size(-1) == embed
    
#     ans = Q.matmul(transpose(K)) 
#     ans /= embed ** 0.5

#     if softmax:
#         ans = ans.softmax(dim=-1)

#     if V is None:
#         return ans

#     assert V.size(-2) == odim
#     return ans.matmul(V)


# class Attention(torch.nn.Module):
#     def __init__(self, dimq, dimk, dimv, head_dim, head=1, embed_dim=None):
#         super(Attention, self).__init__()

#         if embed_dim is None:
#             assert dimq == dimk
#             embed_dim = dimk

#         self.head = head
#         self.head_dim = head_dim
#         self.linearQ = nn.Linear(dimq, embed_dim * head, bias=False)
#         self.linearK = nn.Linear(dimk, embed_dim * head, bias=False)
#         self.linearV = nn.Linear(dimv, head_dim * head, bias=False)


#     def forward(self, Q, K, V):
#         def split_head(M):
#             M = M.reshape(*M.shape[:-1], self.head, self.head_dim)
#             M = swap_dim(M, -2, -3)
#             return M

#         def merge_head(M):
#             M = swap_dim(M, -2, -3)
#             M = M.reshape(*M.shape[:-2], self.head * self.head_dim)
#             return M

#         Q = split_head(self.linearQ(Q))
#         K = split_head(self.linearK(K))
#         V = split_head(self.linearV(V))
#         ans = merge_head(attention(Q, K, V))
        
#         return ans

class Alignment(torch.nn.Module):
    def __init__(self, dim, encode_dims, decode_dims):
        super(Alignment, self).__init__()
        if encode_dims[0] != dim:
            encode_dims = [dim] + encode_dims

        if decode_dims[0] != encode_dims[-1]:
            decode_dims = [encode_dims[-1]] + decode_dims

        if decode_dims[-1] != dim ** 2:
            decode_dims.append(dim ** 2)

        self.dim = dim
        self.encoder = MLP(encode_dims)
        self.decoder = MLP(decode_dims)

    def forward(self, points):
        assert self.dim == points.shape[-1]
        features, _ = self.encoder(points).max(dim=-2)
        affine = self.decoder(features)
        affine = affine.view(*affine.shape[:-1], self.dim, self.dim) + torch.eye(self.dim).cuda()

        # print(points.shape)
        # print(affine.shape)

        return points[:, :, None, :].matmul(affine[:, None, :, :]).squeeze(-2)



class EncoderLayer(torch.nn.Module):

    def __init__(self, ind, layer_type, idim, odim, sample_dim=None, dropout=0.5, relu_weight=0.005):
        super(EncoderLayer, self).__init__()

        self.ind = ind
        self.layer_type = layer_type
        self.idim = idim
        self.odim = odim
        if layer_type == 'leaf':
            assert idim == 3
            self.point_to_embed = MLP([3, odim, odim], init=relu_weight)
        else:
            ndir = build_tree.num_directions()
            
            self.upload = MLP([idim, odim, odim], init=relu_weight)

            if layer_type == 'sampled':
                self.upload_sample = MLP([sample_dim, idim, odim], init=relu_weight)

            
            if ind in [2, 4, 6]:
                self.align = Alignment(odim, [128, 256, 1024], [512, 256])
            else:
                self.align = lambda x : x
            

        self.dropout = lambda x : x # nn.Dropout(dropout)


    def forward(self, last, sample=None, vec=None, dmap=None, drev=None):
        if self.layer_type[0] == 'l':
            ans = self.point_to_embed(last)
        else:
            # ans = self.merge(torch.cat([last[:, self.child_l], last[:, self.child_r]], dim=-1))

            lch = self.upload(last[:, self.child_l])
            rch = self.upload(last[:, self.child_r])
            ans = torch.maximum(lch, rch)

            if self.layer_type[0] == 's':
                smp = self.upload_sample(sample[:, self.child_s])
                ans = torch.maximum(ans, smp)

            ans = self.align(ans)

        return self.dropout(ans)


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

        self.align_pts = Alignment(3, [64, 128, 1024], [512, 256])

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

            sample_dim = None
            if layer_type == 'sampled':
                sample_dim = odims[-self.sample_layers]

            odims.append(odim)
            data = layer_dict[i]

            layer = EncoderLayer(len(self.layers), layer_type, idim, odim, 
                sample_dim=sample_dim, 
                relu_weight=0.25,
                dropout=min(0.1 * (1.5 ** i), 0.4)
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

    def arrange(self, pts):
        pts = self.align_pts(pts)

        batches = pts.detach().cpu()

        inputs = [[] for _ in self.layers]

        for batch in batches:
            _, output, _ = self.tree.arrange(batch, basic=False, rotate=False, debug=False) 
            assert len(output) == len(inputs)
            for i, line in enumerate(output):
                inputs[i].append(line)

        for i, line in enumerate(inputs):
            inputs[i] = torch.stack(line, dim=0).cuda()

        return pts, inputs


    def forward(self, ans, inputs=None, perm=None):

        layer_output = []        
        ans = ans.cuda()

        if perm is not None:
            axisperm, axissgn, dmap, drev = build_tree.transforms[perm]
            ans = ans[:, :, axisperm] * axissgn
        else:
            axisperm = axissgn = dmap = drev = None

        if inputs is None:
            ans, inputs = self.arrange(ans)
        
        for i, (line, layer) in enumerate(zip(inputs, self.layers)):

            vec = None
            if layer.layer_type[0] != 'l':
                vec = line.cuda()
            else:
                # ans : batch_size, numpoints, featuredim
                # arrange: batch_size, layersize
                arrange = line.cuda().unsqueeze(-1).expand(line.shape + (ans.shape[-1], ))
                ans = ans.gather(1, arrange)
                
            sample = None
            if layer.layer_type[0] == 's':
                sample = layer_output[-self.sample_layers]

            ans = layer(ans, sample=sample, vec=vec, dmap=dmap, drev=drev)
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




                

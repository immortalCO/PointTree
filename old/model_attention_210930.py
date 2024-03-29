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


def transpose(ans, dim=-2):
    if dim != -1:
        perm = list(range(len(ans.shape)))
        perm[-1], perm[dim] = perm[dim], perm[-1]
        ans = ans.permute(*perm)

    return ans

class FC(torch.nn.Module):
    def __init__(self, idim, odim, init=0.25, dim=-1, flatten=None):
        super(FC, self).__init__()
        self.dim = dim
        self.flatten = flatten
        self.idim = idim
        self.odim = odim
        self.linear = nn.Linear(idim, odim)
        self.activate = nn.PReLU(init=init)

    def forward(self, ans, dim=None):
        if dim is None:
            dim = self.dim
        ans = transpose(ans, dim=dim)

        if self.flatten is not None:
            shape = ans.shape
            fl = len(self.flatten)
            ans = ans.view(*shape[:-fl], self.idim)


        ans = self.activate(self.linear(ans))

        if self.flatten is not None:
            ans = ans.view(*shape[:-fl], *self.flatten)

        ans = transpose(ans, dim=dim)
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
    def __init__(self, dimq, dimk, embed=None, output_convert=None):
        super(Attention, self).__init__()

        if embed is None:
            assert dimq == dimk
            embed = dimk

        self.linearQ = nn.Linear(dimq, embed)
        self.linearK = nn.Linear(dimk, embed)

        if output_convert is not None:
            self.linearV = nn.Linear(*output_convert)
        else:
            self.linearV = lambda x : x

    def forward(self, Q, K, V):
        return attention(self.linearQ(Q), self.linearK(K), self.linearV(V))

class EncoderLayer(torch.nn.Module):

    def __init__(self, layer_type, channel, dim, dropout=0.5, relu_weight=0.25, input_dim=0, sample_dim=0):
        super(EncoderLayer, self).__init__()

        self.layer_type = layer_type
        self.channel = channel
        self.dim = dim
        self.sample_dim = dim


        if layer_type == 'leaf':
            self.point_to_embed = FC(3, channel * dim, init=relu_weight)
        else:
            # self.save = FC(channel * 2, channel, init=relu_weight, dim=-2)
            # self.save = FC(input_dim * channel * 4, dim * channel, init=relu_weight, flatten=(channel, dim))
            self.attention_l = Attention(input_dim, input_dim, embed=dim)
            self.attention_r = Attention(input_dim, input_dim, embed=dim)
            self.merge = FC(input_dim * 4, dim, init=relu_weight)
            # self.merge = FC(input_dim * 2, dim, init=relu_weight)
            
            if layer_type == 'sampled':
                self.upload = FC(channel * sample_dim, channel * dim, init=relu_weight, flatten=(channel, dim))
                self.attention_ans = Attention(dim, dim)
                self.attention_smp = Attention(dim, dim)
                self.merge_sample = FC(dim * 4, dim, init=relu_weight)

            
        self.dropout = nn.Dropout(dropout, inplace=True)
        # self.dropout = lambda x : x


    def forward(self, last, sample=None, vec=None, dmap=None, drev=None):
        if self.layer_type[0] == 'l':
            ans = self.point_to_embed(last)
            ans = ans.view(ans.size(0), ans.size(1), self.channel, self.dim).contiguous()
        else:
            lch = last[:, self.child_l]
            rch = last[:, self.child_r]

            ans = torch.cat([
                lch, self.attention_l(lch, rch, rch),
                rch, self.attention_r(rch, lch, lch),
            ], dim=-1)
            ans = self.merge(ans)

            if self.layer_type[0] == 's':
                smp = sample[:, self.child_s]
                smp = self.upload(smp)

                ans = torch.cat([
                    ans, self.attention_ans(ans, smp, smp),
                    smp, self.attention_smp(smp, ans, ans),
                ], dim=-1)
                ans = self.merge_sample(ans)

            
        return self.dropout(ans)



class Encoder(torch.nn.Module):

    def __init__(self, N, sample_layers, dim, OUTPUT, channel=1, rotate=True):
        super(Encoder, self).__init__()

        self.OUTPUT = OUTPUT
        self.num_layers = -1
        self.layers = None
        self.channel = channel
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
        dim_repeat = 3
        dims = [0] * (self.sample_layers * dim_repeat)
        cur_dim = 4

        for i in reversed(range(self.num_layers)):
            layer_type = 'sampled'
            if i + self.sample_layers >= self.num_layers:
                layer_type = 'unsampled'
            if i + 1 == self.num_layers:
                layer_type = 'leaf'

            if layer_type != 'leaf' and (len(dims) - 1) % dim_repeat == 0:
                cur_dim = min(dim, cur_dim << 1)
            dims.append(cur_dim)

            data = layer_dict[i]
            layer = EncoderLayer(layer_type, channel, cur_dim, 
                relu_weight=0.25,
                dropout=min(0.1 * (1.3 ** i), 0.4),
                input_dim=dims[-(1 + 1)],
                sample_dim=dims[-(self.sample_layers + 1)]
            )
            layer.num_nodes = len(data)
            data = sorted(data.values(), key=lambda x : x[0])

            if layer_type != 'leaf':
                layer.child_l = torch.tensor([l for _, _, l, _, _ in data]).cuda()
                layer.child_r = torch.tensor([r for _, _, _, r, _ in data]).cuda()

                if layer_type == 'sampled':
                    layer.child_s = torch.tensor([s for _, _, _, _, s in data]).cuda()

            self.layers.append(layer)

            logging.info(f"layer {i} ({layer_type}) # = {layer.num_nodes} dim = {layer.dim}")

        self.layers = nn.ModuleList(self.layers)


    def forward(self, inputs, perm=0):

        backup = []        

        # ignore ind
        ans = None

        d = build_tree.get_directions()
        
        axisperm, axissgn, dmap, drev = build_tree.transforms[perm]

        for i, (line, layer) in enumerate(zip(inputs, self.layers)):

            vec = None
            if layer.layer_type[0] != 'l':
                vec = line.cuda()
            else:
                ans = line.cuda()[:, :, axisperm] * axissgn

            sample = None
            if layer.layer_type[0] == 's':
                sample = backup[-self.sample_layers]


            ans = layer(ans, sample=sample, vec=vec, dmap=dmap, drev=drev)
            backup.append(ans)

        return ans.squeeze(1)


def debug_main():
    OUTPUT = './scratch'
    encoder = Encoder(2048, 2, 128, OUTPUT).cuda()
    for name, param in encoder.named_parameters():
        print(f"Parameter: {name} {param.size()}")

    
if __name__ == '__main__':
    debug_main()




                

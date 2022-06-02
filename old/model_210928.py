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



class EncoderLayer(torch.nn.Module):

    def __init__(self, layer_type, idim, odim, sample_dim=None, relu_weight=0.005):
        super(EncoderLayer, self).__init__()

        self.layer_type = layer_type
        self.idim = idim
        self.odim = odim
        if layer_type == 'leaf':
            assert idim == 3
            self.point_to_embed = nn.Linear(3, odim)
        else:
            ndir = build_tree.num_directions()
            self.merge = nn.ModuleList([nn.Linear(idim * 2, odim) for _ in range(ndir)])
            # self.merge = nn.Linear(idim * 2, odim)
            if layer_type == 'sampled':
                self.merge_sample = nn.Linear(odim + sample_dim, odim)

        self.activate = nn.PReLU(init=relu_weight)
            
        # self.dropout = nn.Dropout(0.5)
        self.dropout = lambda x : x


    def forward(self, last, sample=None, vec=None, dmap=None, drev=None):
        if self.layer_type[0] == 'l':
            ans = self.activate(self.point_to_embed(last))
        else:
            odim = self.odim
            ndir = build_tree.num_directions()

            drev = drev[vec].long()
            lch = last[:, self.child_l]
            rch = last[:, self.child_r]

            ans = torch.cat([
                    lch * (1 - drev)[:, :, None] + rch * drev[:, :, None], 
                    rch * (1 - drev)[:, :, None] + lch * drev[:, :, None]
                ], dim=-1)
            batch_size = ans.size(0)
            layer_size = ans.size(1)
            # (batch_size, layer_size, idim * 2)

            # ans = self.activate(self.merge(ans))

            vec = dmap[vec]
            tmp = torch.zeros(batch_size, layer_size, odim).cuda()
            for i, mer in enumerate(self.merge):
                flag = (vec == i)
                tmp[flag] += mer(ans[flag])
            ans = self.activate(tmp)

            if self.layer_type[0] == 's':
                ans = self.activate(self.merge_sample(torch.cat([ans, sample[:, self.child_s]], dim=-1)))
        return self.dropout(ans)



class Encoder(torch.nn.Module):

    def __init__(self, N, sample_layers, dim, OUTPUT, rotate=True):
        super(Encoder, self).__init__()

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
        odims = []
        for i in reversed(range(self.num_layers)):
            layer_type = 'sampled'
            if i + self.sample_layers >= self.num_layers:
                layer_type = 'unsampled'
            if i + 1 == self.num_layers:
                layer_type = 'leaf'


            if layer_type != 'leaf':
                idim = feature_dim
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

            layer = EncoderLayer(layer_type, idim, odim, sample_dim=sample_dim, relu_weight=0.32 * (0.7 ** i))
            layer.num_nodes = len(data)
            data = sorted(data.values(), key=lambda x : x[0])

            if layer_type != 'leaf':
                layer.child_l = torch.tensor([l for _, _, l, _, _ in data]).cuda()
                layer.child_r = torch.tensor([r for _, _, _, r, _ in data]).cuda()

                if layer_type == 'sampled':
                    layer.child_s = torch.tensor([s for _, _, _, _, s in data]).cuda()

            self.layers.append(layer)

            logging.info(f"layer {i} ({layer_type}) # = {layer.num_nodes} relu_weight = {0.32 * (0.7 ** i)}")

        self.layers = nn.ModuleList(self.layers)

        self.perm3 = []
        for pid, p in enumerate([
                [0, 1, 2], [0, 2, 1],
                [1, 0, 2], [1, 2, 0],
                [2, 0, 1], [2, 1, 0]
            ]):

            mapping = []
            revs = []
            for d in build_tree.get_directions():
                d = d[p]
                k = -1
                rev = None
                for j, d2 in enumerate(build_tree.get_directions()):
                    if (d - d2).norm() < 1e-6 or (d + d2).norm() < 1e-6:
                        k = j
                        rev = (d + d2).norm() < 1e-6
                assert k != -1
                mapping.append(k)
                revs.append(rev)

            self.perm3.append([torch.tensor(p).cuda(), torch.tensor(mapping).cuda(), torch.tensor(revs).cuda()])

    def directions(self):
        d = build_tree.get_directions().clone()
        return d

    def forward(self, inputs, perm=0):

        backup = []        

        # ignore ind
        ans = None

        d = self.directions()
        
        axisperm, dmap, drev = self.perm3[perm]

        for i, (line, layer) in enumerate(zip(inputs, self.layers)):

            vec = None
            if layer.layer_type[0] != 'l':
                vec = line.cuda()
            else:
                ans = line.cuda()[:, :, axisperm]

            sample = None
            if layer.layer_type[0] == 's':
                sample = backup[-self.sample_layers]

            # print(f"forward #{i} ans = {ans.shape} vec = {vec if vec is None else vec.shape}")

            ans = layer(ans, sample=sample, vec=vec, dmap=dmap, drev=drev)
            backup.append(ans)

        return ans.squeeze(-2)


def debug_main():
    OUTPUT = './scratch'
    encoder = Encoder(2048, 2, 128, OUTPUT).cuda()
    for name, param in encoder.named_parameters():
        print(f"Parameter: {name} {param.size()}")

    
if __name__ == '__main__':
    debug_main()




                

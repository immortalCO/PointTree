import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging

def init_logging(OUTPUT):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s:\t%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(f"{OUTPUT}/training.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

class EncoderLayer(torch.nn.Module):

    def __init__(self, layer_type, dim):
        super(EncoderLayer, self).__init__()

        self.layer_type = layer_type
        if layer_type == 'leaf':
            self.point_to_embed = nn.Linear(3, dim)
        else:
            self.merge_children = nn.Linear(dim * 2, dim)
            if layer_type == 'sampled':
                self.merge_sample = nn.Linear(dim * 2, dim)

        self.activate = nn.PReLU()
            
        self.dropout = nn.Dropout(0.5)


    def forward(self, last, sample=None):
        if self.layer_type[0] == 'l':
            ans = self.activate(self.point_to_embed(last))
        else:
            ans = self.activate(self.merge_children(torch.cat([last[:, self.child_l], last[:, self.child_r]], dim=-1)))
            if self.layer_type[0] == 's':
                ans = self.dropout(ans)
                ans = self.activate(self.merge_sample(torch.cat([ans, sample[:, self.child_s]], dim=-1)))
        return self.dropout(ans)



class Encoder(torch.nn.Module):

    def __init__(self, N, sample_layers, dim, OUTPUT):
        super(Encoder, self).__init__()

        self.OUTPUT = OUTPUT
        self.num_layers = -1
        self.layers = None
        self.dim = dim

        layer_dict = None

        # generate tree structure
        # os.system("g++ build_tree.cpp -o build_tree -O2 -std=c++14 -Wall")
        os.system(f"./build_tree {sample_layers} struct {N} > {OUTPUT}/struct.txt")
        with open(f"{OUTPUT}/struct.txt") as file:
            cur_layer = -1

            def pmap(layer, p):
                p = int(p)
                if layer < 0 or p == -1:
                    return [-1]
                
                if p not in layer_dict[layer]:
                    layer_dict[layer][p] = [len(layer_dict[layer])]
                return layer_dict[layer][p]

            for line in file:
                line = line.split(' ')
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

        for i in reversed(range(self.num_layers)):
            layer_type = 'sampled'
            if i + self.sample_layers >= self.num_layers:
                layer_type = 'unsampled'
            if i + 1 == self.num_layers:
                layer_type = 'leaf'

            data = layer_dict[i]
            layer = EncoderLayer(layer_type, self.dim)
            layer.num_nodes = len(data)
            data = sorted(data.values(), key=lambda x : x[0])

            if layer_type != 'leaf':
                layer.child_l = torch.tensor([l for _, _, l, _, _ in data]).cuda()
                layer.child_r = torch.tensor([r for _, _, _, r, _ in data]).cuda()

                if layer_type == 'sampled':
                    layer.child_s = torch.tensor([s for _, _, _, _, s in data]).cuda()

            self.layers.append(layer)

            logging.info(f"layer {i} ({layer_type}) # = {layer.num_nodes}")

        self.layers = nn.ModuleList(self.layers)

    def forward(self, ans):
        ans = ans.cuda()

        backup = []
        for layer in self.layers:
            sample = None
            if layer.layer_type[0] == 's':
                sample = backup[-self.sample_layers]
            ans = layer(ans, sample)
            backup.append(ans)

        ans.squeeze_(dim=-2)
        return ans


def debug_main():
    OUTPUT = './scratch'
    encoder = Encoder(2048, 2, 128, OUTPUT).cuda()
    for name, param in encoder.named_parameters():
        print(f"Parameter: {name} {param.size()}")

    
if __name__ == '__main__':
    debug_main()




                

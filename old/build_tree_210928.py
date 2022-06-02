import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import random
import logging

global directions
PI = torch.acos(torch.tensor(-1.0))
directions = None
otho = []
basic_directions = set()

def init_directions():
    global directions
    if directions is not None:
        return num_directions()

    directions = []
    for x in [0, 1]:
        for y in [-1, 0, 1]:
            if x == 0 and y < 0:
                continue
            for z in [-1, 0, 1]:
                if x == y == 0 and z < 0:
                    continue

                if 0 < abs(x) + abs(y) + abs(z):
                    d = torch.tensor([x, y, z]).float().cuda()
                    d /= d.norm()
                    directions.append(d)
                    otho.append(set())

                if abs(x) + abs(y) + abs(z) == 1:
                    basic_directions.add(len(directions) - 1)

    logging.info(f"init_directions: # = {len(directions)}")
    for i, d in enumerate(directions):
        
        for j, e in enumerate(directions):
            if d.dot(e).abs().item() < 1e-6:
                otho[i].add(j)

        logging.debug(f"i: {d} #otho = {len(otho[i])}")

    logging.info(f"basic # = {len(basic_directions)}")

    directions = torch.stack(directions, dim=0)
    return num_directions()

def num_directions():
    global directions
    if directions is None:
        init_directions()
    return directions.size(0)

def get_directions():
    global directions
    if directions is None:
        init_directions()
    return directions



def split_tensor(dist, device='cuda'):
    n = dist.size(0) >> 1
    cut, _ = dist.kthvalue(n)

    ind = torch.arange(n * 2).to(device)
    lind = ind[dist < cut]
    rind = ind[dist > cut]
    mind = ind[dist == cut]

    if lind.size(0) != n:
        lind = torch.cat([lind, mind[: +(n - lind.size(0))]], dim=0)

    if rind.size(0) != n:
        rind = torch.cat([mind[-(n - rind.size(0)) :], rind], dim=0)

    return lind, rind


def solve(pts, fixed, basic_lim=-1, random_lim=64, cuda_lim=64):

    device = 'cpu' if pts.size(0) <= cuda_lim else 'cuda'
    pts = pts.to(device)

    def split(pts, vec):
        prod = (pts * vec).sum(dim=-1)
        lind, rind = split_tensor(prod, device=device)
        pls = pts[lind]
        prs = pts[rind]
        dist = prs.mean() - pls.mean() + ((pls.std() + prs.std()) if pts.size(0) > 2 else 0)
        return dist
        
    vec = list(range(num_directions()))
    if fixed is not None:
        vec = list(set(vec) & otho[fixed])

    if pts.size(0) <= random_lim:
        return random.choice(vec), 0

    if pts.size(0) <= basic_lim:
        vec = list(set(vec) & basic_directions)
    
    random.shuffle(vec)
    vals = [split(pts, get_directions()[v].to(device)) for v in vec]
    cho = torch.argmax(torch.tensor(vals)).item()

    return cho, vals[cho]


class TreeNode:
    def __init__(self, l, r, s):
        self.l = l
        self.r = r
        self.s = s

class BuildTree:

    def __init__(self, N, sample_layers):
        while (N & -N) != N:
            ++N
        self.N = N
        self.sample_layers = sample_layers
        self.num_directions = init_directions()

    def structure(self):
        N = self.N
        sample_layers = self.sample_layers
        mem = []
        layers = [[] for i in range(len(bin(N)) - 2)]

        def build_struct(npts, dep):
            p = len(mem)
            mem.append(None)
            layers[dep].append(p)

            if npts == 1:
                mem[p] = TreeNode(-1, -1, -1)
            else:
                l = build_struct(npts >> 1, dep + 1)
                r = build_struct(npts >> 1, dep + 1)
                s = -1

                if (npts >> sample_layers) > 0:
                    s = build_struct(npts >> sample_layers, dep + sample_layers)

                mem[p] = TreeNode(l, r, s)

            return p

        build_struct(N, 0)
        output = []
        output.append(('size', self.N))
        output.append(('sample', self.sample_layers))
        for i, layer in reversed(list(enumerate(layers))):
            if len(layer) == 0:
                continue
            output.append(('layer', i))
            for p in layer:
                pp = mem[p]
                output.append(('node', p, -1, pp.l, pp.r, pp.s))

        return output

    def arrange(self, pts):
        # pts: array n * 3

        N = self.N
        sample_layers = self.sample_layers

        if len(pts) < N:
            from random import randint
            tn = len(pts)
            while len(pts) < N:
                k = randint(0, tn)
                ind.append(k)
                pts.append(pts[k])

        pts = pts.cuda()
        ind = torch.arange(len(pts)).cuda()

        output = [[] for i in range(len(bin(N)) - 2)]
        arrange = []
        # print(len(output))

        counter = [0]

        def build(ind, fixed):
            dep = len(bin(ind.size(0))) - 3

            if ind.size(0) == 1:
                arrange.append(ind)
                output[dep].append(ind)
                return

            def split(ind, vec):
                prod = (pts[ind] * vec).sum(dim=-1)
                lind, rind = split_tensor(prod)
                return lind, rind

            vec, loss = solve(pts[ind], fixed)
            output[dep].append(vec)

            counter[0] += 1
            # if counter[0] <= 5 or counter[0] % 1000 == 0:
            #       logging.debug(f"build #{counter[0]} |ind| = {ind.size(0)} fixed? = {fixed is not None} vec = {vec} loss = {loss}")

            d = get_directions()[vec]
            ls, rs = split(ind, d)
            build(ind[ls], None)
            build(ind[rs], None)

            if (ind.size(0) >> sample_layers) > 0:
                n = ind.size(0)
                ns = n >> sample_layers
                choice = torch.randperm(n)[:ns]
                build(ind[choice], vec)

        build(ind, None)

        for i, line in enumerate(output):
            output[i] = torch.tensor(line).cuda()
        
        return output, torch.tensor(arrange).cuda()



if __name__ == '__main__':

    b = BuildTree(2048, 2)
    structure = b.structure()
    for line in structure:
        print(" ".join(map(str, line)))



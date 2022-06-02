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
transforms = []

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

                if 0 < abs(x) + abs(y) + abs(z) <= 3:
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

        logging.debug(f"{i}: {d} otho = {otho[i]}")

    logging.info(f"basic # = {len(basic_directions)}")

    directions = torch.stack(directions, dim=0)

    for pid, p in enumerate([
            [0, 1, 2], [0, 2, 1],
            [1, 0, 2], [1, 2, 0],
            [2, 0, 1], [2, 1, 0]
        ]):

        mapping = []
        revs = []
        for d in directions:
            d = d[p]
            k = -1
            rev = None
            for j, d2 in enumerate(directions):
                if (d - d2).norm() < 1e-6 or (d + d2).norm() < 1e-6:
                    k = j
                    rev = (d + d2).norm() < 1e-6
            assert k != -1
            mapping.append(k)
            revs.append(rev)

        transforms.append([torch.tensor(p).cuda(), torch.tensor(mapping).cuda(), torch.tensor(revs).cuda()])
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
    _, sind = dist.sort(dim=-1)
    ls = sind[:dist.size(0) >> 1]
    rs = sind[dist.size(0) >> 1:]
    return ls, rs

def solve(pts, fixed):

    dep = len(bin(pts.size(0))) - 3

    return dep % num_directions(), 0


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
        self.cpp_compiled = False
        self.layer_size = None

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
        self.layer_size = list(map(len, layers))

        for i, layer in reversed(list(enumerate(layers))):
            if len(layer) == 0:
                continue
            output.append(('layer', i))
            for p in layer:
                pp = mem[p]
                output.append(('node', p, -1, pp.l, pp.r, pp.s))

        return output

    def arrange_cpp(self, pts, complete_output=True):
        from os import system
        if not self.cpp_compiled:
            self.cpp_compiled = True
            system("g++ build_tree.cpp -o /tmp/build_tree -O3 -Wall")
        if self.layer_size is None:
            self.structure()

        with open(f"/tmp/cppinput.txt", 'w') as file:
            file.write("%d\n" % (num_directions()))
            for d in get_directions():
                file.write("%lf %lf %lf\n" % tuple(d.cpu().numpy().tolist()))
            for p in pts.cpu().numpy().tolist():
                file.write("%lf %lf %lf\n" % tuple(p))

        system(f"/tmp/build_tree {self.sample_layers} arrange {self.N} < /tmp/cppinput.txt > /tmp/cppoutput.txt")

        output = []
        with open(f"/tmp/cppoutput.txt") as file:
            for i, line in enumerate(file):
                if line == '':
                    continue
                line = tuple(map(int, line.split()))
                assert len(line) == self.layer_size[-i - 1]
                output.append(torch.tensor(line).cuda())

        return output, output[0]


    def arrange(self, pts):
        # pts: array n * 3

        return self.arrange_cpp(pts)

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



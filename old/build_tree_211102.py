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

def init_directions(samples=100, calc_dmap=False):
    import math
    global directions
    if directions is not None:
        return num_directions()

    lim = 1 / (samples ** 0.5)

    directions = []

    phi = math.pi * (3. - math.sqrt(5.))
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        directions.append([x, y, z])
        otho.append(set([]))

    directions = torch.tensor(directions).cuda()


    logging.info(f"init_directions: # = {len(directions)}")
    for i, d in enumerate(directions):
        
        dot = (directions * d).sum(dim=-1).abs()
        cho = dot <= lim

        otho[i] = set(torch.arange(dot.shape[0])[cho].cpu().numpy().tolist())

        logging.debug(f"{i}: {' '.join(map(lambda x : '%.6lf' % x, d.cpu().numpy().tolist()))} otho = {otho[i]}")

    logging.info(f"basic # = {len(basic_directions)}")

    for pid, p in enumerate([
            [0, 1, 2], [0, 2, 1],
            [1, 0, 2], [1, 2, 0],
            [2, 0, 1], [2, 1, 0]
        ]):

        for sgnset in range(1 << 3):
            sgn = torch.tensor(list(map(lambda x : -1 if x == '1' else 1, bin(sgnset + 8)[3:]))).cuda()

            mapping = []
            revs = []

            if calc_dmap:
                for od in directions:
                    d = od[p] * sgn
                    k = -1
                    rev = None
                    for j, d2 in enumerate(directions):
                        if (d - d2).norm() < 1e-6 or (d + d2).norm() < 1e-6:
                            k = j
                            rev = ((d + d2).norm() < 1e-6).item()
                    assert k != -1
                    mapping.append(k)
                    revs.append(rev)
            else:
                ndir = len(directions)
                mapping = list(range(ndir))
                revs = [False] * ndir

            # logging.info(f"transform #{len(transforms)} {p} {sgn.cpu().numpy().tolist()} {mapping} {revs}")
            transforms.append([torch.tensor(p).cuda(), sgn.cuda(), torch.tensor(mapping).cuda(), torch.tensor(revs).cuda()])

    logging.info(f"transforms # = {len(transforms)}")
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

PI = torch.acos(torch.tensor(-1.0))

def otho_vector(vec):
    x, y, z = vec
    if abs(x) >= max(abs(y), abs(z)):
        ret = torch.FloatTensor([-y - z, x, x]).to(vec.device)
    elif abs(y) >= max(abs(x), abs(z)):
        ret = torch.FloatTensor([y, -x - z, y]).to(vec.device)
    else:
        ret = torch.FloatTensor([z, z, -x - y]).to(vec.device)
    ret /= ret.norm()
    return ret

def rotate_towards(pts, vec, dim=3):
    # rotate one of the axis to vec
    vec /= vec.norm()
    if dim == 2:
        assert vec[2].abs() < 1e-5

        vx, vy, _ = vec.split([1, 1, 1], dim=-1)
        x, y, z, = pts.split([1, 1, 1], dim=-1)
        x, y = x * vx + y * vy, x * vy - y * vx

        return torch.cat([x, y, z], dim=-1).to(pts.device)

    with torch.no_grad():
        z_axis = vec
        y_axis = otho_vector(z_axis)
        x_axis = z_axis.cross(y_axis)
        y_axis /= y_axis.norm()
        x_axis /= x_axis.norm()

    x = (pts * x_axis).sum(dim=-1, keepdim=True)
    y = (pts * y_axis).sum(dim=-1, keepdim=True)
    z = (pts * z_axis).sum(dim=-1, keepdim=True)

    return torch.cat([x, y, z], dim=-1).to(pts.device)


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

    def arrange(self, pts, basic=False, debug=False, rotate=True, rotate_only=False, extra=False, num_rotate=10, vec_per_point=4, device='cpu'):
        from os import system
        if not self.cpp_compiled:
            self.cpp_compiled = True
            system("g++ build_tree.cpp -o tmp/build_tree -O3 -Wall")
            system("g++ build_tree_basic.cpp -o tmp/build_tree_basic -O3 -Wall")
            system("g++ build_tree_extra.cpp -o tmp/build_tree_extra -O3 -Wall")
        if self.layer_size is None:
            self.structure()

        n = pts.size(0)

        # find the best axis
        pts -= pts.mean(dim=0)
        pts /= pts.abs().mean()
        
        debug_print = print if debug else lambda *args, **kwargs : 0

        if rotate:
            
            flag_z_count = 0

            for i_rotate in range(num_rotate):
                debug_print(i_rotate)
                pts = pts.cuda()
                amx = (pts[None, :, :] - pts[:, None, :]).norm(dim=-1).argmax().item()
                z_axis = pts[amx // n] - pts[amx % n]
                z_axis /= z_axis.norm()

                debug_print("z_axis", format(z_axis))
                flag_z = (z_axis - torch.tensor([0, 0, 1]).cuda()).norm() < 1e-6

                if flag_z:
                    flag_z_count += 1
                    if flag_z_count > 1:
                        break
                else:
                    flag_z_count = 0

                pts = rotate_towards(pts, z_axis)
                z_axis = rotate_towards(z_axis, z_axis)
                debug_print("z_axis z rotated", format(z_axis))

                z_axis = torch.tensor([0., 0., 1.]).cuda()
                
                amx = (pts[None, :, :-1] - pts[:, None, :-1]).norm(dim=-1).argmax().item()
                x_axis = pts[amx // n] - pts[amx % n]
                x_axis[-1] = 0
                x_axis /= x_axis.norm()
                debug_print("x_axis", format(x_axis))

                pts = rotate_towards(pts, x_axis, dim=2)
                z_axis = rotate_towards(z_axis, x_axis, dim=2)
                x_axis = rotate_towards(x_axis, x_axis, dim=2)
                debug_print("z_axis x rotated", format(z_axis))
                debug_print("x_axis x rotated", format(x_axis))

                if rotate_only:
                    break

                for i in range(3):
                    pts[:, i] -= pts[:, i].mean()
                    pts[:, i] /= pts[:, i].abs().mean() 
            
            
        pts = pts.to(device)


        # arrange with cpp
        with open(f"tmp/cppinput.txt", 'w') as file:
            if not basic:
                file.write("%d\n" % (num_directions()))
                for d in get_directions():
                    file.write("%lf %lf %lf\n" % tuple(d.cpu().numpy().tolist()))
            for p in pts.cpu().numpy().tolist():
                file.write("%lf %lf %lf\n" % tuple(p))

        suffix = ''
        suffix += '_basic' if basic else ''
        suffix += '_extra' if extra else ''


        from os import chdir
        chdir("tmp")
        btcpp = "build_tree" + suffix
        debug_print(f"{btcpp} {self.sample_layers} arrange {self.N} < cppinput.txt > cppoutput.txt")
        assert 0 == system(f"{btcpp} {self.sample_layers} arrange {self.N} < cppinput.txt > cppoutput.txt")
        chdir('..')

        output = []
        with open(f"tmp/cppoutput.txt") as file:
            num_layers = len(self.layer_size)
            file = list(file)
            for i, line in enumerate(file[:num_layers]):
                line = tuple(map(int, line.split()))
                if not basic or len(output) == 0:
                    assert len(line) == self.layer_size[-i - 1]
                output.append(torch.tensor(line).cuda())

            if extra:
                extra_features = []
                for i, line in enumerate(file[num_layers:num_layers + self.N]):
                    extra_features.append(tuple(map(float, line.split())))

                extra_features = torch.tensor(extra_features).float()


        return pts, output, extra_features if extra else output[0] 



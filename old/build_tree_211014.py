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

def init_directions(chaos_limit=3):
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

                if 0 < abs(x) + abs(y) + abs(z) <= chaos_limit:
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

        for sgnset in range(1 << 3):
            sgn = torch.tensor(list(map(lambda x : -1 if x == '1' else 1, bin(sgnset + 8)[3:]))).cuda()

            mapping = []
            revs = []

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

        return torch.cat([x, y, z], dim=-1)

    with torch.no_grad():
        z_axis = vec
        y_axis = otho_vector(z_axis)
        x_axis = z_axis.cross(y_axis)
        y_axis /= y_axis.norm()
        x_axis /= x_axis.norm()

    x = (pts * x_axis).sum(dim=-1, keepdim=True)
    y = (pts * y_axis).sum(dim=-1, keepdim=True)
    z = (pts * z_axis).sum(dim=-1, keepdim=True)

    return torch.cat([x, y, z], dim=-1)

class Vector(torch.nn.Module):
    def __init__(self, fixed=None, device='cpu'):
        super(Vector, self).__init__()
        self.fixed = fixed
        self.device = device

        if fixed is None:
            self.u = nn.Parameter(torch.tensor([random.uniform(0, 1)]))
            self.v = nn.Parameter(torch.tensor([random.uniform(0, 1)]))
        else:
            x, y, z = fixed
            self.axis = torch.FloatTensor([x, y, z]).to(device)
            self.axis /= self.axis.norm()
            self.base = otho_vector(self.axis)
            assert (self.base * self.axis).sum().abs().item() < 1e-5

            self.u = nn.Parameter(torch.tensor([random.uniform(0, 1)]))

    def forward(self):
        if self.fixed is None:
            u = self.u.fmod(1)
            v = self.v.fmod(1)
            # assert not u.isnan().item()
            # assert not v.isnan().item()

            theta = 2 * PI.to(self.device) * u
            phi = torch.acos((2 * v - 1).clamp(min=-1.0, max=1.0))
            sin_phi = torch.sin(phi)
            x = torch.cos(theta) * sin_phi
            y = torch.sin(theta) * sin_phi
            z = torch.cos(phi)
        else:
            u = self.u.fmod(1)
            theta = 2 * PI.to(self.device) * u
            ax, ay, az = self.axis 
            bx, by, bz = self.base
            Sin = torch.sin(theta)
            Cos = torch.cos(theta)
            x = bx + ay.pow(2)*bx*(-1 + Cos) + az.pow(2)*bx*(-1 + Cos) + ax*ay*(by - by*Cos) + ax*az*(bz - bz*Cos) - az*by*Sin + ay*bz*Sin
            y = ay*(ax*bx + ay*by + az*bz) - (ax*ay*bx + (-1 + ay.pow(2))*by + ay*az*bz)*Cos + (az*bx - ax*bz)*Sin
            z = az*(ax*bx + ay*by + az*bz) - (ax*az*bx + ay*az*by + (-1 + az.pow(2))*bz)*Cos + (-(ay*bx) + ax*by)*Sin

        vec = torch.stack([x.squeeze(), y.squeeze(), z.squeeze()], dim=0)
        vec = vec / vec.norm()

        if vec.isnan().sum().item() != 0:
            if self.fixed is None:
                print(u, v, theta, phi)
            else:
                print(self.axis, self.base, u, theta, Sin, Cos)

            assert False

        return vec


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

    def arrange(self, pts, basic=False, debug=False, rotate=True, vec_per_point=4, device='cpu'):
        from os import system
        if not self.cpp_compiled:
            self.cpp_compiled = True
            system("g++ build_tree.cpp -o /tmp/build_tree -O3 -Wall")
            system("g++ build_tree_basic.cpp -o /tmp/build_tree_basic -O3 -Wall")
        if self.layer_size is None:
            self.structure()

        n = pts.size(0)

        # find the best axis
        pts = pts.cuda()
        amx = (pts[None, :, :] - pts[:, None, :]).norm(dim=-1).argmax().item()
        axis_z = pts[amx // n] - pts[amx % n]
        axis_z /= axis_z.norm()

        rotate_towards(pts, axis_z)
        rotate_towards(axis_z, axis_z)

        amx = (pts[None, :, :-1] - pts[:, None, :])




        
        pts = pts.to(device)

        if rotate:
            debug_print = print if debug else lambda *args, **kwargs : 0

            points = pts.cuda()
            allvec = points - points[:, None]
            alldist = allvec.norm(dim=-1)
            alldist[alldist < 1e-8] += 1e10

            _, ind = alldist.topk(vec_per_point, dim=-1, largest=False)
            ind = ind[:, 1 :]
            ind = torch.stack([ind] * 3, dim=-1)
            surf = allvec.gather(1, ind)
            surf = surf / surf.norm(dim=-1, keepdim=True)
            assert surf.isnan().sum().item() == 0
            surface = surf.to(device)      
            del points, allvec, alldist, ind, surf

            def train(axis):
                num_epoch = 10
                opt = torch.optim.Adam(axis.parameters(), lr=1e-2)

                for epoch in range(num_epoch):
                    vec = axis()

                    def side_metric(d, window_size=1):
                        return 0.
                        # return d.std()
                        # scale = (d[-1] - d[0]).detach()
                        # d = (d[window_size:] - d[:-window_size]) / scale
                        # return -(d - d.mean()).tanh().mean()

                    def overall_metric(vec):
                        dotprod = (vec * surface).sum(dim=-1)
                        dotprod = dotprod.abs()
                        mark = dotprod.min(1 - dotprod)
                        return -mark.mean()

                    def calc(pts, vec):
                        # assert vec.isnan().sum().item() == 0
                        dist = (pts * vec).sum(dim=-1)
                        ls, rs = split_tensor(dist)
                        ld, rd = dist[ls], dist[rs]
                        loss = 0.
                        loss -= (rd.mean() - ld.mean()).abs()
                        # assert loss.isnan().sum().item() == 0

                        loss -= overall_metric(vec)
                        # assert loss.isnan().sum().item() == 0

                        loss -= side_metric(ld) + side_metric(rd)
                        # assert loss.isnan().sum().item() == 0

                        return loss, ls, rs

                    loss, _, _ = calc(pts, vec)
                    # debug_print(f"train #{epoch} loss = {loss.item()}")
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                return axis().detach(), loss.item()

            def format(tensor):
                return ", ".join(map(lambda x : "%.6lf"%x, tensor.cpu().numpy().tolist()))

            def multiple_train(new_model, num_trains=10):
                axis = None
                loss = 1e10
                for t in range(num_trains):
                    a, l = train(new_model())
                    if l < loss:
                        axis, loss = a, l
                return axis, loss

            z_axis, z_loss = multiple_train(lambda : Vector(device=device))
            debug_print("z_loss %.8lf" % (z_loss))
            debug_print("z_axis", format(z_axis))

            pts = rotate_towards(pts, z_axis)
            z_axis = rotate_towards(z_axis, z_axis)
            debug_print("z_axis z rotated", format(z_axis))

            z_axis = torch.tensor([0., 0., 1.]).to(device)
            x_axis, x_loss = multiple_train(lambda : Vector(z_axis, device=device))
            debug_print("x_loss %.8lf" % (x_loss))
            debug_print("x_axis", format(x_axis))

            pts = rotate_towards(pts, x_axis, dim=2)
            z_axis = rotate_towards(z_axis, x_axis, dim=2)
            x_axis = rotate_towards(x_axis, x_axis, dim=2)
            debug_print("z_axis x rotated", format(z_axis))
            debug_print("x_axis x rotated", format(x_axis))
        

        # arrange with cpp
        with open(f"/tmp/cppinput.txt", 'w') as file:
            if not basic:
                file.write("%d\n" % (num_directions()))
                for d in get_directions():
                    file.write("%lf %lf %lf\n" % tuple(d.cpu().numpy().tolist()))
            for p in pts.cpu().numpy().tolist():
                file.write("%lf %lf %lf\n" % tuple(p))

        basic = '_basic' if basic else ''
        system(f"/tmp/build_tree{basic} {self.sample_layers} arrange {self.N} < /tmp/cppinput.txt > /tmp/cppoutput.txt")

        output = []
        with open(f"/tmp/cppoutput.txt") as file:
            for i, line in enumerate(file):
                if line == '':
                    continue
                line = tuple(map(int, line.split()))
                if not basic or len(output) == 0:
                    assert len(line) == self.layer_size[-i - 1]
                output.append(torch.tensor(line).cuda())

        return pts, output, output[0]



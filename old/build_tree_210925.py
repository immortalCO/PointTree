import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import random
import logging

PI = torch.acos(torch.tensor(-1.0))

def rotate_towards(pts, vec):
    # rotate the x-axis to vec
    
    def axis2sphere(vec):
        r = vec.norm(dim=-1, keepdim=True)
        x, y, z = (vec / r).split([1, 1, 1], dim=-1)
        phi = torch.acos(z)
        the = torch.atan2(y, x)
        return r, the, phi

    _, vthe, vphi = axis2sphere(vec)
    r, the, phi = axis2sphere(pts)
    
    the = the + vthe
    phi = phi + vphi

    sin_phi = phi.sin()
    x = r * torch.cos(the) * sin_phi
    y = r * torch.sin(the) * sin_phi
    z = r * torch.cos(phi)
   
    return torch.cat([x, y, z], dim=-1)


class Vector(torch.nn.Module):
    def __init__(self, fixed, device='cpu'):
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

            if abs(x) >= max(abs(y), abs(z)):
                self.base = torch.FloatTensor([-y - z, x, x]).to(device)
            elif abs(y) >= max(abs(x), abs(z)):
                self.base = torch.FloatTensor([y, -x - z, y]).to(device)
            else:
                self.base = torch.FloatTensor([z, z, -x - y]).to(device)
            self.base /= self.base.norm()

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
            # assert not u.isnan().item()

            theta = 2 * PI.to(self.device) * u
            ax = self.axis[0]
            ay = self.axis[1]
            az = self.axis[2]
            bx = self.base[0]
            by = self.base[1]
            bz = self.base[2]
            Sin = torch.sin(theta)
            Cos = torch.cos(theta)
            x = bx + ay.pow(2)*bx*(-1 + Cos) + az.pow(2)*bx*(-1 + Cos) + ax*ay*(by - by*Cos) + ax*az*(bz - bz*Cos) - az*by*Sin + ay*bz*Sin
            y = ay*(ax*bx + ay*by + az*bz) - (ax*ay*bx + (-1 + ay.pow(2))*by + ay*az*bz)*Cos + (az*bx - ax*bz)*Sin
            z = az*(ax*bx + ay*by + az*bz) - (ax*az*bx + ay*az*by + (-1 + az.pow(2))*bz)*Cos + (-(ay*bx) + ax*by)*Sin

        # assert not x.isnan().item()
        # assert not y.isnan().item()
        # assert not z.isnan().item()

        vec = torch.stack([x.squeeze(), y.squeeze(), z.squeeze()], dim=0)
        vec = vec / vec.norm()
        return vec


def solve(pts, fixed, fast_lim=128, num_epoch=30, check_epoch=2, lim=1e-4):

    check_epoch += 1

    if pts.size(0) <= fast_lim:
        # vec = Vector(fixed, 'cpu').cpu()
        # return vec().detach().cuda(), 1
        vec = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        return torch.tensor(vec[random.randint(0, 2)]).float().cuda(), 0


    def split(pts, vec):
        prod = (pts * vec).sum(dim=-1)
        _, sind = prod.sort(dim=-1)
        ls = sind[:pts.size(0) >> 1]
        rs = sind[pts.size(0) >> 1:]
        return prod[ls], prod[rs]

    device = 'cuda'
    pts = pts.to(device)
    vector = Vector(fixed, device).to(device)
    opt = torch.optim.Adam(vector.parameters(), lr=2e-2)

    history = []

    for epoch in range(num_epoch):
        vec = vector()

        history.append(vec.detach())
        if len(history) >= check_epoch and (history[-check_epoch] - history[-1]).norm() < lim:
            break

        pls, prs = split(pts, vec)
        loss = -( prs.mean() - pls.mean() + ((pls.std() + prs.std()) if pts.size(0) > 2 else 0) )

        # if loss.isnan().item():
        #   print(pls, pls.mean(), pls.std())
        #   print(prs, prs.mean(), prs.std())
        # assert not loss.isnan().item()

        opt.zero_grad()
        loss.backward()
        opt.step()

    return vector().detach().cuda(), loss.item()

# def solve(pts, fixed):
#   vec = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
#   dep = len(bin(pts.size(0)))
#   return torch.tensor(vec[dep % 3]).float().cuda(), 0

# def solve(pts, fixed, dup=5):

#   device = 'cuda'

#   pts = pts.to(device)
#   perm = torch.randperm(pts.size(0)).to(device)
#   set0 = torch.cat([torch.randperm(pts.size(0)).to(device) for _ in range(dup)], dim=0)
#   set1 = (set0 + 1) % pts.size(0)
    
#   vec = pts[perm[set0]] - pts[perm[set1]]
#   dist = vec.norm(dim=-1)

#   cut = -(-dist).kthvalue(max(1, min(128, pts.size(0))))[0]

#   vec = vec[dist >= cut].sum(dim=0)
#   vec /= vec.norm()

#   return vec.cuda(), 0



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

        def build(curpts, ind, fixed):
            dep = len(bin(ind.size(0))) - 3

            # print(f"build size = {curpts.size()} fixed? = {fixed is not None}")
            # assert curpts.size(0) == ind.size(0)
            # assert curpts.size(1) == 3

            if ind.size(0) == 1:
                # print(dep)
                output[dep].append(curpts.squeeze(-2))
                arrange.append(ind)
                return

            def split(ind, vec):
                prod = (curpts * vec).sum(dim=-1)
                _, sind = prod.sort(dim=-1)
                ls = sind[:ind.size(0) >> 1]
                rs = sind[ind.size(0) >> 1:]
                return ls, rs

            vec, loss = solve(curpts, fixed)
            output[dep].append(vec)

            counter[0] += 1
            # if counter[0] <= 5 or counter[0] % 1000 == 0:
            #       logging.debug(f"build #{counter[0]} |ind| = {ind.size(0)} fixed? = {fixed is not None} vec = {vec} loss = {loss}")

            ls, rs = split(ind, vec)
            downpts = rotate_towards(curpts, vec)
            build(downpts[ls], ind[ls], None)
            build(downpts[rs], ind[rs], None)

            if (ind.size(0) >> sample_layers) > 0:
                n = ind.size(0)
                ns = n >> sample_layers
                choice = torch.randperm(n)[:ns]
                build(curpts[choice], ind[choice], vec)

        build(pts, ind, None)

        assert len(output[-1]) != 0

        for i, line in enumerate(output):
            output[i] = torch.stack([v for v in line], dim=0)
        
        return output, torch.tensor(arrange).cuda()



if __name__ == '__main__':

    b = BuildTree(2048, 2)
    structure = b.structure()
    for line in structure:
        print(" ".join(map(str, line)))



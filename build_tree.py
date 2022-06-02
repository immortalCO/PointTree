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

def init_directions(chaos_limit=4, calc_dmap=True):
    global directions
    if directions is not None:
        return num_directions()

    basic_only = False
    if chaos_limit == 0:
        basic_only = True
        chaos_limit = 1

    directions = []

    # for x in [0, 1]:
    #     for y in [-1, 0, 1]:
    #         if x == 0 and y < 0:
    #             continue
    #         for z in [-1, 0, 1]:
    #             if x == y == 0 and z < 0:
    #                 continue

    #             if 0 < abs(x) + abs(y) + abs(z) <= chaos_limit:
    #                 d = torch.tensor([x, y, z]).float()
    #                 d /= d.norm()
    #                 directions.append(d)
    #                 otho.append(set())

    #             if abs(x) + abs(y) + abs(z) == 1:
    #                 basic_directions.add(len(directions) - 1)

    for x in reversed(range(-chaos_limit, chaos_limit + 1)):
        for y in reversed(range(-chaos_limit, chaos_limit + 1)):
            for z in reversed(range(-chaos_limit, chaos_limit + 1)):

                # print(x, y, z)
                # if x < 0:
                #     continue
                # if abs(x) < 1e-6 and y < 0:
                #     continue
                # if abs(x) < 1e-6 and abs(y) < 1e-6 and z < 0:
                #     continue

                d = torch.tensor([x, y, z]).float()

                if d.norm() < 1e-6:
                    continue

                d /= d.norm()

                for d2 in directions:
                    if (d - d2).norm() < 1e-6 or (d + d2).norm() < 1e-6:
                        d = None
                        break

                if d is None:
                    continue

                directions.append(d)
                otho.append(set())

                if abs(abs(x) + abs(y) + abs(z) - 1) <= 1e-5:
                    basic_directions.add(len(directions) - 1)



    if basic_only:
        directions = [directions[i] for i in basic_directions]

    logging.info(f"init_directions: # = {len(directions)}")
    for i, d in enumerate(directions):
        
        for j, e in enumerate(directions):
            if d.dot(e).abs().item() < 1e-6:
                otho[i].add(j)

        logging.debug(f"{i}: {' '.join(map(lambda x : '%.6lf' % x, d.cpu().numpy().tolist()))} otho = {otho[i]}")

    logging.info(f"basic # = {len(basic_directions)}")

    directions = torch.stack(directions, dim=0)

    for pid, p in enumerate([
            [0, 1, 2], [0, 2, 1],
            [1, 0, 2], [1, 2, 0],
            [2, 0, 1], [2, 1, 0]
        ]):

        for sgnset in range(1 << 3):
            sgn = torch.tensor(list(map(lambda x : -1 if x == '1' else 1, bin(sgnset + 8)[3:])))

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


def split_tensor(dist, device='cpu'):
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


def symmetry_loss(pts, vec):
    n = pts.size(0)
    pts = rotate_towards(pts, vec)
    dotprod = pts[:, -1]

    _, lch_ind = dotprod.topk(k=n // 2, largest=False, sorted=False)
    lch = torch.zeros(n).bool()
    lch[lch_ind] = True
    rch = ~lch

    lch = pts[lch]
    rch = pts[rch]

    # cut = (lch[:, -1].max() + rch[:, -1].min()).item() / 2
    cut = pts.mean(dim=0)[-1].item()
    lch[:, -1] = cut - lch[:, -1]
    rch[:, -1] = rch[:, -1] - cut
   
    # dist = (lch[:, None, :] - rch[None, :, :]).norm(dim=-1)
    # dist = torch.cat([dist, dist.T], dim=-1)
    # dist = dist.min(dim=-1)[0]

    # loss = dist.clamp(min=1e-6).pow(-1).mean().pow(-1)
    # return loss.item(), dotprod

    vals = [
        lambda p : ((p[:, 0] - 10) * 20 + (p[:, 1] - 10)) * 20 + (p[:, 2] - 10),
        lambda p : ((p[:, 0] - 10) * 20 + (p[:, 2] - 10)) * 20 + (p[:, 1] - 10),
        lambda p : ((p[:, 1] - 10) * 20 + (p[:, 0] - 10)) * 20 + (p[:, 2] - 10),
        lambda p : ((p[:, 1] - 10) * 20 + (p[:, 2] - 10)) * 20 + (p[:, 0] - 10),
        lambda p : ((p[:, 2] - 10) * 20 + (p[:, 1] - 10)) * 20 + (p[:, 0] - 10),
        lambda p : ((p[:, 2] - 10) * 20 + (p[:, 0] - 10)) * 20 + (p[:, 1] - 10),
    ]

    def invperm(p):
        return torch.empty_like(p).scatter_(0, p, torch.arange(len(p), device=p.device))

    scale = pts.max(dim=0).values - pts.min(dim=0).values

    linds = torch.stack([ val(lch) for val in vals ], dim=0).sort(dim=-1).indices
    rinds = torch.stack([ val(rch) for val in vals ], dim=0).sort(dim=-1).indices

    rch_sort = []
    for lind, rind in zip(linds, rinds):
        rch_sort.append(rch[rind][invperm(lind)])
    rch_sort = torch.stack(rch_sort, dim=0)

    diff = ((lch - rch_sort) / scale).norm(dim=-1).min(dim=0)[0]

    # return diff.clamp(min=1e-6).pow(-1).mean().pow(-1)
    loss = (diff < 0.1).sum() / pts.size(0)
    return -loss.item(), dotprod

def density_scale(pts, debug=False):
    from scipy.spatial import Delaunay
    pts = pts.clone()
    pts -= pts.mean(dim=0)
    pts /= pts.abs().mean()
    tri_ind = torch.tensor(Delaunay(pts.cpu().numpy()).simplices).long()

    mask = torch.ones(tri_ind.size(0)).bool()
    for i in range(4):
        for j in range(i + 1, 4):
            lc = pts[tri_ind[:, i]]
            rc = pts[tri_ind[:, j]]
            num = lc.size(0)
            norms = (lc - rc).norm(dim=-1)
            last_mask_sum = mask.sum()
            for _ in range(16):
                thres = norms[mask].mean() * 3 - norms[mask].min()
                mask &= (norms <= thres)
                mask_sum = mask.sum()
                if last_mask_sum == mask_sum:
                    continue
                last_mask_sum = mask_sum

    tri_ind = tri_ind[mask]

    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            edges.append(tri_ind[:, [i, j]])

    edges = torch.cat(edges, dim=0)

    edge_diff = (pts[edges[:, 0]] - pts[edges[:, 1]]).abs()
    edge_diff /= edge_diff.mean()
    mask = edge_diff >= 1e-3 + 1e-5
    scale = (edge_diff * mask).sum(dim=0) / mask.sum(dim=0)
    if debug:
        print(f"scale = {scale}")
    return pts / scale

class TreeNode:
    def __init__(self, l, r, s):
        self.l = l
        self.r = r
        self.s = s

class BuildTree:

    def __init__(self, N, sample_layers, sample_child_first=True, sample_cross=False, use_symmetry_loss=False, record_vec=False):
        while (N & -N) != N:
            ++N
        self.N = N
        self.depth_lim = len(bin(N)) - 3
        self.sample_layers = sample_layers
        self.num_directions = init_directions()
        self.cpp_compiled = False
        self.layer_size = None
        self.use_symmetry_loss = use_symmetry_loss
        self.dir_symloss = get_directions()
        
        self.sample_cross = sample_cross
        if sample_cross and not sample_child_first:
            logging.info("`sample_child_first` overridden due to `sample_cross`")
            sample_child_first = True

        self.sample_child_first = sample_child_first
        # print("build_tree init depth_lim =", self.depth_lim)

        self._upsample_tree = None
        if self.N > 2048:
            self.upsample_tree()

        self.use_sym = use_symmetry_loss
        self.record_vec = record_vec

    def upsample_tree(self):
        if self._upsample_tree is None:
            self._upsample_tree = BuildTree(self.N // 2, self.sample_layers, 
                sample_child_first=self.sample_child_first, sample_cross=self.sample_cross, use_symmetry_loss=False)
        return self._upsample_tree

    def structure(self):
        N = self.N
        sample_layers = self.sample_layers
        mem = []
        layers = [[] for i in range(self.depth_lim + 1)]

        def build_struct(npts, dep):
            p = len(mem)
            mem.append(None)
            layers[dep].append(p)

            if npts == 1 or dep == self.depth_lim:
                mem[p] = TreeNode(-1, -1, -1)
            else:

                if not self.sample_child_first:
                    l = build_struct(npts >> 1, dep + 1)
                    r = build_struct(npts >> 1, dep + 1)

                # modification: (l, r, s) -> (s, l, r)
                if not self.sample_cross and (npts >> sample_layers) > 0 and (dep + sample_layers <= self.depth_lim):
                    s = build_struct(npts >> sample_layers, dep + sample_layers)
                else:
                    s = -1

                if self.sample_child_first:
                    l = build_struct(npts >> 1, dep + 1)
                    r = build_struct(npts >> 1, dep + 1)

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

    def arrange_pca(self, pts):
        N = self.N

        ind = torch.arange(len(pts))

        if len(pts) < N and not self.sample_cross:
            from random import randint
            tn = len(pts)
            while len(pts) < N:
                k = randint(0, tn)
                ind.append(k)
                pts.append(pts[k])

        pts = pts.cpu()
        cnt = torch.zeros(pts.shape[0])

        output = [[] for _ in self.layer_size]

        def build(ind, dep, fixed=None):
            from random import choice

            n = len(ind)

            if n == 1 or dep == self.depth_lim:

                app_cnt = cnt[ind]
                ind = ind[app_cnt == app_cnt.min().item()]
                cho = choice(ind).item()
                cnt[cho] += 1

                output[dep].append(cho)
                return

            U, _, V = torch.pca_lowrank(pts[ind])
            if fixed is None:
                dotprod = U.T[0]
                vec = V.T[0]
            else:
                dotval = 1000
                for _dotprod, _vec in zip(U.T, V.T):
                    _dotval = fixed.dot(_vec).abs()
                    if dotval > _dotval + 1e-5:
                        dotval = _dotval
                        dotprod = _dotprod
                        vec = _vec

            if self.use_symmetry_loss:
                axis = vec.abs().max(dim=-1)[1]
                dotprod = pts[ind][:, axis]
                vec = torch.tensor([0., 0., 0.])
                vec[axis] = 1.

            if random.randint(0, 1) == 1:
                dotprod = -dotprod
                vec = -vec

            output[dep].append(vec.sgn().char())

            lch_val, lch_ind = dotprod.topk(k=n // 2, largest=False, sorted=False)
            mask = torch.zeros_like(ind).bool()
            mask[lch_ind] = True

            lch = mask
            rch = ~mask

            if self.sample_cross and (n >> self.sample_layers) > 0:
                rch_val = dotprod[rch]
                rch_ind = torch.arange(n)[rch]

                s = (n >> self.sample_layers)
                lch[ rch_ind[rch_val.topk(k=s, largest=False, sorted=False).indices] ] = True
                rch[ lch_ind[lch_val.topk(k=s, largest=True,  sorted=False).indices] ] = True

                del rch_val
                del rch_ind

            del lch_val
            del lch_ind
            del dotprod


            if not self.sample_child_first:
                build(ind[lch], dep + 1)
                build(ind[rch], dep + 1)

            if not self.sample_cross and (n >> self.sample_layers) > 0 and (dep + self.sample_layers <= self.depth_lim):
                s = n >> self.sample_layers
                cho = torch.randperm(n)[:s]
                build(ind[cho], dep + self.sample_layers, vec)
            

            if self.sample_child_first:
                build(ind[lch], dep + 1)
                build(ind[rch], dep + 1)


        build(ind, 0)

        output = output[::-1]
        output[0] = torch.tensor(output[0])
        for i in range(1, len(output)):
            output[i] = torch.stack(output[i], dim=0)
        return pts, output, torch.tensor([[0.0]])


    def arrange(self, pts, pca=True, basic=False, debug=False, rotate=True, rotate_only=False, extra=False, num_rotate=10, vec_per_point=4, device='cpu'):
        from os import system
        if not self.cpp_compiled:
            self.cpp_compiled = True
            # system("g++ build_tree.cpp -o tmp/build_tree -O3 -Wall")
            # system("g++ build_tree_basic.cpp -o tmp/build_tree_basic -O3 -Wall")
            # system("g++ build_tree_extra.cpp -o tmp/build_tree_extra -O3 -Wall")
        if self.layer_size is None:
            self.structure()

        pts, extra = pts.split([3, pts.size(-1) - 3], dim=-1)

        n = pts.size(0)
        pts = pts.cpu()

        center = pts.mean(dim=0)
        pts -= center
        pts /= pts.norm(dim=-1).mean()
        pts += center
        
        debug_print = print if debug else lambda *args, **kwargs : 0
        debug_print(f"arrange n = {n} self.N = {self.N}")

        if rotate:
            if rotate_only:
                pts, v, _ = torch.pca_lowrank(pts)
                pts = pts.mm(v.diag())
            else:
                # pts, _, _ = torch.pca_lowrank(pts)

                # iterative
                cvg_count = 0
                for i_rot in range(10):
                    pts, _, V = torch.pca_lowrank(pts)
                    z_axis = V.T[0]

                    if (z_axis.abs() - torch.tensor([0., 0., 1.])).norm() < 1e-6:
                        cvg_count += 1
                    else:
                        cvg_count = 0
                    if cvg_count > 1:
                        break

                    pts -= pts.mean(dim=0)
                    for i in range(3):
                        pts[:, i] /= pts[:, i].abs().mean() 

            pts -= pts.mean(dim=0)
            pts /= pts.norm(dim=-1).mean()

        if not self.record_vec:
            import tree_builder_cpp
            debug_print("call cpp: " + ("arrange" if self.use_sym else "arrange_no_sym"))
            arrange = (tree_builder_cpp.arrange if self.use_sym else tree_builder_cpp.arrange_no_sym)
            arrange = arrange(self.N, pts.cpu().tolist())
            debug_print("cpp return");

            output = [torch.tensor([0]) for _ in self.layer_size]
            output[0] = torch.tensor(arrange)

            return torch.cat([pts, extra], dim=-1), output, torch.tensor([[0.0]])

        else:
            return self.arrange_pca(pts)
            
            
        # pts = pts.to(device)

        # if n <= self.N // 2:
        #     pts, output = self.upsample_tree().arrange(pts, pca=pca, debug=debug, 
        #         rotate=False, rotate_only=False, extra=False, device='cpu')[:2]
        #     arrange = output[0]
        #     assert len(arrange) == self.N // 2

        #     ind = torch.randint_like(arrange, 0, self.N // 4)
        #     coe = torch.rand(arrange.shape)

        #     p1 = pts[arrange[ind << 1]]
        #     p2 = pts[arrange[ind << 1 | 1]]
        #     newpts = p1 * coe[:, None] + p2 * (1 - coe)[:, None]
        #     debug_print(f"newpts = {newpts.shape}")

        #     pts = torch.cat([pts, newpts], dim=0)
        #     assert len(pts) == self.N


        # # if pca:
        # return self.arrange_pca(pts)

        # assert not self.sample_child_first

        # # arrange with cpp
        # with open(f"tmp/cppinput.txt", 'w') as file:
        #     if not basic:
        #         file.write("%d\n" % (num_directions()))
        #         for d in get_directions():
        #             file.write("%lf %lf %lf\n" % tuple(d.cpu().numpy().tolist()))
        #     for p in pts.cpu().numpy().tolist():
        #         file.write("%lf %lf %lf\n" % tuple(p))

        # suffix = ''
        # suffix += '_basic' if basic else ''
        # suffix += '_extra' if extra else ''


        # from os import chdir
        # chdir("tmp")
        # btcpp = "build_tree" + suffix
        # debug_print(f"{btcpp} {self.sample_layers} arrange {self.N} < cppinput.txt > cppoutput.txt")
        # assert 0 == system(f"{btcpp} {self.sample_layers} arrange {self.N} < cppinput.txt > cppoutput.txt")
        # chdir('..')

        # output = []
        # with open(f"tmp/cppoutput.txt") as file:
        #     num_layers = len(self.layer_size)
        #     file = list(file)
        #     for i, line in enumerate(file[:num_layers]):
        #         line = tuple(map(int, line.split()))
        #         if not basic or len(output) == 0:
        #             assert len(line) == self.layer_size[-i - 1]
        #         output.append(torch.tensor(line))

        #     if extra:
        #         extra_features = []
        #         for i, line in enumerate(file[num_layers:num_layers + self.N]):
        #             extra_features.append(tuple(map(float, line.split())))

        #         extra_features = torch.tensor(extra_features).float()


        # return pts.cpu(), output, extra_features if extra else output[0] 


def dynamic_arrange(pts, ind=True, pca=False, device='cuda'):
    batch = pts.shape[0]
    if ind:
        pts = pts.detach()
        ind = torch.arange(pts.shape[1], device=device)[None, None, :].expand(batch, 1, -1)
    pts = pts[:, None, :, :]

    while pts.shape[2] > 1:
        batch, node, sub, dim = pts.shape

        if pca:
            val, _, _ = torch.pca_lowrank(pts, q=3)
        else:
            axis = pts.var(dim=-2).argmax(dim=-1)
            val = pts.gather(dim=-1, index=axis[:, :, None, None].expand(-1, -1, sub, dim))[:, :, :, 0]

        topk = (val.topk(sub // 2, dim=-1).indices + 1).to_sparse_coo()
        mask = torch.sparse_coo_tensor(indices=torch.cat([topk.indices()[:2], topk.values()[None, :] - 1], dim=0), 
                        values=torch.ones(batch * node * sub // 2, dtype=torch.bool, device=device), size=(batch, node, sub)).to_dense()

        lch = pts[mask].reshape(batch, node, sub // 2, dim)
        rch = pts[~mask].reshape(batch, node, sub // 2, dim)

        # lvi = val.topk(sub // 2, largest=False).indices
        # rvi = val.topk(sub // 2, largest=True).indices

        # lch = pts.gather(2, lvi[:, :, :, None].expand(-1, -1, -1, dim))
        # rch = pts.gather(2, rvi[:, :, :, None].expand(-1, -1, -1, dim))
        
        pts = torch.cat([lch, rch], dim=1)

        if ind is not False:
            lind = ind[mask].reshape(batch, node, sub // 2)
            rind = ind[~mask].reshape(batch, node, sub // 2)
            
            # lind = ind.gather(2, lvi)
            # rind = ind.gather(2, rvi)
            
            ind = torch.cat([lind, rind], dim=1)

    if ind is not False:
        return ind.squeeze(2)
    return pts.squeeze(2)

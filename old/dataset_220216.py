import torch
from random import uniform, randint
from math import acos, cos, sin, log, pi
import copy
import h5py
import numpy as np
import logging



def transform_generator(points, scale=False, rotate=False, rotate_single_axis=False, affine=False, homo=False):
    if scale:
        scale = torch.randn(3)
        points = points * scale

    if rotate:
        ox = uniform(0, 2 * pi)
        oy = uniform(0, 2 * pi)
        oz = uniform(0, 2 * pi)

        if rotate_single_axis is not False:
            if rotate_single_axis is True:
                k = randint(1, 3)
            else:
                k = rotate_single_axis
            if k != 1:  ox = 0.
            if k != 2:  oy = 0.
            if k != 3:  oz = 0.

        ox = torch.tensor([
            [1, 0, 0],
            [0, cos(ox), -sin(ox)],
            [0, sin(ox), cos(ox)]
        ]).float()
        oy = torch.tensor([
            [cos(oy), 0, -sin(oy)],
            [0, 1, 0],
            [sin(oy), 0, cos(oy)],
        ]).float()
        oz = torch.tensor([
            [cos(oz), -sin(oz), 0],
            [sin(oz), cos(oz), 0],
            [0, 0, 1]
        ]).float()
        axisperm = torch.randperm(3)
        axissgn = torch.tensor(-1).pow(torch.randint(low=0, high=2, size=[3]))
        points = (points.matmul(ox).matmul(oy).matmul(oz))[:, axisperm] * axissgn

    if affine is not False:
        if affine is True:
            affine = torch.nn.Linear(3, 3, bias=False).weight.detach()
            points = points.matmul(affine)
        else:
            def simple_align(points):
                return points
                # return torch.pca_lowrank(points)[0]

            n = points.shape[0]
            A = torch.tensor([])
            B = torch.tensor([])
            C = torch.tensor([])
            m = n // 2
            while len(A) < m:
                randperm = lambda : torch.randint(0, n, [n])
                A = randperm()
                B = randperm()
                C = randperm()
                mask = (A != B) & (B != C) & (A != C)
                A = A[mask]
                B = B[mask]
                C = C[mask]

            A = A[:m]
            B = B[:m]
            C = C[:m]

            def angles(points):
                l1 = points[B] - points[A]
                l2 = points[C] - points[A]
                l1 /= l1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                l2 /= l2.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                return (l1 * l2).sum(dim=-1).clamp(min=-1, max=1).acos()

            orig_angles = angles(simple_align(points))

            def angle_diff(points):
                diff = angles(simple_align(points)) - orig_angles
                diff = diff.abs()
                diff = torch.minimum(diff, 2 * torch.pi - diff)
                return diff.mean()

            ans = 0
            best_M = None
            for _ in range(3000):
                M = torch.nn.Linear(3, 3, bias=False).weight.detach()

                diff = angle_diff(points.matmul(M))

                if diff > ans:
                    ans = diff
                    best_M = M

                if ans > affine:
                    break
                affine *= 0.9995

            points = points.matmul(best_M)


    if homo:
        homo = torch.nn.Linear(4, 4, bias=False).weight.detach()
        hpts = torch.cat([points, torch.ones([points.size(0), 1])], dim=-1).matmul(homo)
        scale = hpts[:, -1:]
        scale_clamp_min = log(1. / 4.)
        scale_clamp_max = log(4.)
        scale_sgn = scale.sign()
        scale = scale.abs().log()
        scale -= scale.min()
        scale /= scale.max()
        scale = scale * (scale_clamp_max - scale_clamp_min) + scale_clamp_min
        scale = scale.exp() * scale_sgn
        points = hpts[:, :-1] / scale

    return points


def make_data_generator(points, arrange, transform=lambda x : x, pca=True, rotate=True, rotate_only=False, extra=True, debug=False):
    points = transform(points)
    points, output, extra = arrange(points, basic=False, rotate=rotate, extra=extra, pca=pca,
                                    rotate_only=rotate_only, debug=debug)
    
    for j, val in enumerate(output):
        assert val.min().item() >= 0
        vmax = val.max().item()
        if vmax < 128:
            output[j] = val.char().cpu()
        else:
            assert vmax < 65536
            output[j] = val.short().cpu()

    points = points.cpu()
    extra = extra.cpu()

    # assert 0 <= label.min().item() <= label.max().item() < 10000
    return (points, output, extra, None)

def make_data_default(points, arrange, transform=lambda x : x):
    return make_data_generator(points, arrange, transform, pca=True)

def make_data_rotate_only(points, arrange, transform=lambda x : x):
    return make_data_generator(points, arrange, transform, pca=True, rotate=True, rotate_only=True)

def make_data_no_prealign(points, arrange, transform=lambda x : x):
    return make_data_generator(points, arrange, transform, pca=True, rotate=False, rotate_only=False)

def no_transform(pts):
    return pts

def affine_transform(pts):
    return transform_generator(pts, affine=True)

class affine_lim:
    def __init__(self, lim):
        self.lim = lim
    def __call__(self, pts):
        return transform_generator(pts, affine=self.lim)

def placeholder(x):
    return x

class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, clouds, labels, arrange, augment=1, make=make_data_default, transform=lambda x : x, extra_labels=None, subset=None):
        self.transform = transform
        if subset is not None:
            self.clouds = clouds[subset]
            self.labels = labels[subset]
            if extra_labels is not None:
                self.extra_labels = extra_labels[subset]
            else:
                self.extra_labels = None

        else:
            self.clouds = clouds
            self.labels = labels
            self.extra_labels = extra_labels

        self.make = make
        self.augment = augment
        self.arrange = arrange
        self.mem = None

    def __len__(self):
        return self.clouds.shape[0] * self.augment
    
    def __getitem__(self, i):
        if self.mem is not None:
            data = list(self.mem[i])[:-1]
            data.append(self.labels[i // self.augment].cpu())
            if self.extra_labels is not None:
                data.append(self.extra_labels[i // self.augment].cpu())
            else:
                data.append(None)

            return tuple(data)

        return self.make(self.clouds[i // self.augment], self.arrange, self.transform)

class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, crit):
        self.dataset = dataset
        self.index = []
        for i, data in enumerate(dataset):
            if crit(data):
                self.index.append(i)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        return self.dataset[self.index[i]]


def make_batch(batches, eval=False, dropout=None, augment=False):
    from random import randint
    layers = 0
    points = []
    inputs = None
    labels = []
    extras = []
    extra_labels = []
    for pts, output, extra, label, extra_label in batches:
        # Simple data augment
        # sgn = torch.tensor(-1).pow(torch.randint(low=0, high=2, size=[extra.shape[-1]]))
        
        # pts -= pts.mean(dim=0)
        # pts /= pts.abs().mean()

        # for i in range(3):
        #     pts[:, i] -= pts[:, i].mean()
        #     pts[:, i] /= pts[:, i].abs().mean() 

        if not eval:

            # def randlr(l, r, *size):
            #     return l + (r - l) * torch.rand(size).to(pts.device)

            # scale = randlr(2/3, 3/2, 3)
            # shift = randlr(-0.2, 0.2, 3)
            # pts = pts * scale + shift

            if dropout is not None:
                n = pts.size(0)
                rep = torch.full([n // 2], dropout).bernoulli()
                rep = torch.arange(n // 2)[rep.bool()] << 1
                
                # rep |= torch.randint_like(rep, 0, 1)
                # output[0][rep] = output[0][rep ^ 1]
                
                arrange = output[0].long()
                # coef = torch.rand(rep.shape)
                # val = pts[arrange[rep]] * coef[:, None] + pts[arrange[rep ^ 1]] * (1 - coef)[:, None]
                # pts[arrange[rep]] = val
                # pts[arrange[rep ^ 1]] = val

                def randlr(l, r):
                    return torch.rand(rep.shape) * (r - l) + l

                pts1 = pts[arrange[rep]]
                pts2 = pts[arrange[rep ^ 1]]
                val = lambda coef : pts1 * (1 - coef)[:, None] + pts2 * coef[:, None]
                pts[arrange[rep]], pts[arrange[rep ^ 1]] = val(randlr(-0.1, 1)), val(randlr(0, 1.1))

            if augment:
                base = pts.max(dim=0).values - pts.min(dim=0).values

                shift = torch.randn(3).to(pts.device)
                pts += shift * base * 1e-2

                jitter = torch.randn_like(pts)
                pts += jitter * base * 1e-3

                scale = torch.randn(3).to(pts.device)
                pts *= scale * 1e-2 + 1
                pts = transform_generator(pts, rotate=True, rotate_single_axis=2)

                
            pass
        
        points.append(pts)
        labels.append(label)
        extras.append(extra)

        if extra_label is not None:
            extra_labels.append(extra_label)
        else:
            extra_labels = None

        if inputs is None:
            layers = len(output)
            inputs = [[] for _ in output]

        for line, out in zip(inputs, output):
            line.append(out)

    points = torch.stack(points, dim=0).float()
    extras = torch.stack(extras, dim=0).float()
    for i, line in enumerate(inputs):
        inputs[i] = torch.stack(line, dim=0).long()
    labels = torch.stack(labels, dim=0)

    ret = [(points, inputs, extras), labels]

    if extra_labels is not None:
        extra_labels = torch.stack(extra_labels, dim=0)
        ret.append(extra_labels)
    
    return ret

def make_batch_generator(eval=False, dropout=None, augment=False):
    return lambda batches : make_batch(batches, eval=eval, dropout=dropout, augment=augment)

def make_batch_train(batches):
    return make_batch(batches, eval=False)

def make_batch_eval(batches):
    return make_batch(batches, eval=True)

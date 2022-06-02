import torch
from random import uniform, randint
from math import acos, cos, sin, log, pi
import copy
import h5py
import numpy as np
import logging



def transform_generator(points, scale=False, rotate=False, rotate_single_axis=False, affine=False, homo=False, homo_iter=3, homo_pow=4, homo_mov=0.1):
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

        if rotate_single_axis is False:
            axisperm = torch.randperm(3)
            axissgn = torch.tensor(-1).pow(torch.randint(low=0, high=2, size=[3]))
            points = (points.matmul(ox).matmul(oy).matmul(oz))[:, axisperm] * axissgn
        else:
            points = points.matmul(ox).matmul(oy).matmul(oz)

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

            const = points.abs().mean().item()
            cpts = torch.cat([points, torch.full([n, 1], const)], dim=-1)

            ans = 0
            best_M = None
            for _ in range(3000):
                M = torch.nn.Linear(4, 3, bias=False).weight.T.detach()

                diff = angle_diff(cpts.matmul(M))

                if diff > ans:
                    ans = diff
                    best_M = M

                if ans > affine:
                    break
                affine *= 0.9995

            points = cpts.matmul(best_M)


    if homo:
        def randlr(l, r, size):
            return l + (r - l) * torch.rand(size)

        def randsgn(size):
            return torch.tensor(-1).pow(torch.randint(low=0, high=2, size=size))

        def randpoint(rmin, rmax):
            u = torch.tensor(uniform(0, 1))
            v = torch.tensor(uniform(0, 1))
            r = torch.tensor(uniform(rmin, rmax))

            theta = 2 * torch.pi * u
            phi = torch.acos((2 * v - 1).clamp(min=-1.0, max=1.0))
            sin_phi = torch.sin(phi)
            x = r * torch.sin(theta) * sin_phi
            y = r * torch.cos(theta) * sin_phi
            z = r * torch.cos(phi)

            return torch.stack([x, y, z], dim=-1)

        for _ in range(homo_iter):
            points = transform_generator(points, affine=True)
            points -= points.mean(dim=0)
            points /= points.norm(dim=-1).max().clamp(min=1e-4)
            # points = transform_generator(points, rotate=True)

            homo = torch.ones(4, 4)

            homo[0, 0 : 3] = randpoint(0, 2)
            homo[1, 0 : 3] = randpoint(0, 2)
            homo[2, 0 : 3] = randpoint(0, 2)
            homo[3, 0 : 3] = randpoint(0, 2)

            vert = randpoint(0, 1)
            homo[:3, :] *= vert[:, None]

            dot = (points * vert).sum(dim=-1)
            if dot.min().clamp(max=0).abs() > dot.max().clamp(min=0).abs():
                homo[3, 3] = -dot.min() + (uniform(0, 1) ** homo_pow + homo_mov)
            else:
                homo[3, 3] = -dot.max() - (uniform(0, 1) ** homo_pow + homo_mov)

            hpts = torch.cat([points, torch.ones([points.size(0), 1])], dim=-1).matmul(homo)
            scale = hpts[:, -1:]

            # print(homo, scale.abs().min(), dot.min(), dot.max())
            # if scale.abs().min().item() < 1e-2:
            #     pos = scale.abs().min(dim=0)[1]
            #     print("Error: ", vert, points[pos])

            points = hpts[:, :-1] / scale

    return points


def make_data_generator(points, arrange, transform=lambda x : x, pca=True, rotate=True, rotate_only=False, extra=True, debug=False):
    points[:, :3] = transform(points[:, :3])
    points, output, extra = arrange(points, basic=False, rotate=rotate, extra=extra, pca=pca,
                                    rotate_only=rotate_only, debug=debug)

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

def homo_transform(pts):
    return transform_generator(pts, homo=True)

def rotate_yaxis(pts):
    return transform_generator(pts, rotate=True, rotate_single_axis=2)

class affine_lim:
    def __init__(self, lim):
        self.lim = lim
    def __call__(self, pts):
        return transform_generator(pts, affine=self.lim)

def placeholder(x):
    return x

def augment_generator(pts, fetch_perm=False, dropout=None, shift=False, jitter=False, scale=False, rotate_yaxis=False, agg_coef=1):
    base = (pts[:, :3] - pts[:, :3].mean(dim=0)).norm(dim=-1).mean().item() * agg_coef

    def randlr(l, r, shape):
        return torch.rand(shape) * (r - l) + l

    if fetch_perm:
        perm = torch.arange(pts.size(0))
    else:
        perm = None

    if dropout is not None:
        n = pts.size(0)
        dropout = dropout * torch.rand(1).item()
        # prev: dropout = 0.5
        # curr: dropout = 0.875 * rand01
        mask = torch.full([n], dropout).bernoulli().bool()
        left = torch.arange(n)[~mask]
        repl = left[torch.randint(low=0, high=left.shape[0], size=[mask.sum()])]
        pts[mask] = pts[repl]
        if fetch_perm:
            perm[mask] = perm[repl]

    if scale:
        pts[:, :3] *= randlr(2/3, 3/2, [3])

    if shift:
        pts[:, :3] += randlr(-0.2, 0.2, [3]) * base

    if jitter:
        pts[:, :3] += (torch.randn_like(pts[:, :3]) * 0.01).clamp(min=-0.02, max=0.02) * base

    if rotate_yaxis:
        pts[:, :3] = transform_generator(pts[:, :3], rotate=True, rotate_single_axis=2)

    return pts, perm


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, clouds, labels, arrange, augment=1, make=make_data_default, 
        transform=lambda x : x, extra_labels=None, subset=None, force_online=False, augment_fn=None, sample_points=2048, 
        use_norm=False, trunc=999999999):
        self.transform = transform
        self.force_online = force_online
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
        self.augment_fn = augment_fn
        self.labels_only = False
        self.sample_points = sample_points
        self.use_norm = use_norm
        
        if not use_norm:
            trunc = min(trunc, 3)
        self.clouds = self.clouds[:, :, :trunc]


    def __len__(self):
        return self.clouds.shape[0] * self.augment
    
    def __getitem__(self, i):
        label = self.labels[i // self.augment].cpu()
        if self.extra_labels is not None:
            extra_label = self.extra_labels[i // self.augment].cpu()

        if self.labels_only:
            data = []
        elif self.mem is None:
            cloud = self.clouds[i // self.augment]
            subset = torch.randperm(cloud.shape[0])[:self.sample_points]
            cloud = cloud[subset]

            # if not self.use_norm:
            #     cloud = cloud[:, :3]

            if len(label.shape) > 0 and label.shape[0] == cloud.shape[0]:
                label = label[subset]
            if self.extra_labels is not None and len(extra_label.shape) > 0 and extra_label.shape[0] == cloud.shape[0]:
                extra_label = extra_label[subset]

            if self.augment_fn is not None:
                cloud[:, :3], perm = self.augment_fn(cloud[:, :3])

                if perm is not None:
                    if self.use_norm:
                        cloud[:, 3:] = cloud[perm, 3:]  
                    if len(label.shape) > 0 and label.shape[0] == cloud.shape[0]:
                        label = label[perm]
                    if self.extra_labels is not None and len(extra_label.shape) > 0 and extra_label.shape[0] == cloud.shape[0]:
                        extra_label = extra_label[perm]

            data = self.make(cloud, self.arrange, self.transform)
            if not self.force_online:
                return data
            data = list(data)[:-1]
        else:
            data = list(self.mem[i])[:-1]

        data.append(label)
        if self.extra_labels is not None:
            data.append(extra_label)
        else:
            data.append(None)
        return tuple(data)


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, crit):
        self.dataset = dataset
        self.index = []
        dataset.labels_only = True
        for i, data in enumerate(dataset):
            if crit(data):
                self.index.append(i)
        dataset.labels_only = False

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        return self.dataset[self.index[i]]

class BalanceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, label_pos=-1, coef=8):
        self.dataset = dataset
        dataset.labels_only = True

        count = dict()
        for i, data in enumerate(dataset):
            c = data[label_pos].item()
            if c not in count:
                count[c] = 0
            count[c] += 1

        num_each = max(count.values()) * coef

        self.index = []
        for i, data in enumerate(dataset):
            self.index += [i] * int(num_each / count[data[label_pos].item()] + 0.5)

        dataset.labels_only = False


    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        return self.dataset[self.index[i]]



def make_batch(batches, eval=False, post_augment_fn=lambda x : x, **kwargs):
    from random import randint
    layers = 0
    points = []
    inputs = None
    labels = []
    extras = []
    extra_labels = []
    for pts, output, extra, label, extra_label in batches:

        if not eval:
            pts = post_augment_fn(pts)
        
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
        inputs[i] = torch.stack(line, dim=0)
    labels = torch.stack(labels, dim=0)

    ret = [(points, inputs, extras), labels]

    if extra_labels is not None:
        extra_labels = torch.stack(extra_labels, dim=0)
        ret.append(extra_labels)
    
    return ret

def make_batch_generator(eval=False, dropout=None, augment=False, pca_augment=False):
    return lambda batches : make_batch(batches, eval=eval, dropout=dropout, augment=augment, pca_augment=pca_augment)

def make_batch_train(batches):
    return make_batch(batches, eval=False)

def make_batch_eval(batches):
    return make_batch(batches, eval=True)

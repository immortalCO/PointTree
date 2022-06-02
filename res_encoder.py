import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import build_tree
import random

from encoder import Encoder, MLP, Alignment
from segment import Segment

class ResEncoder(torch.nn.Module):
    def __init__(self, num_layer, carry_dim, *args, point_dim=3, **kwargs):
        super().__init__()

        layers = []
        
        for i in range(num_layer):
            block = Segment(carry_dim, Encoder(*args, **kwargs))

            # lpd = block.layers[-1][0]
            # lpd.layers[-1].bn = nn.Identity()
            # lpd.layers[-1]._bn = lambda x : x

            layers.append(block)
            kwargs['point_dim'] = carry_dim

        self.layers = nn.ModuleList(layers)
        self.feed = Encoder(*args, **kwargs)

        self.tree = self.feed.tree
        self.dim = self.feed.dim
        self.align_dim = self.dim

    def forward(self, ans, *args, **kwargs):
        ans = ans.cuda()
        for i, layer in enumerate(self.layers):
            # print(f"ResEncoder #{i}")

            out = layer(ans, *args, **kwargs)
            self.align_feature = layer.features
            ans = out if i == 0 else ans + out

        # print(f"ResEncoder #feed")
        return self.feed(ans, *args, **kwargs)


class ResSegment(torch.nn.Module):
    def __init__(self, carry_dim, encoder_with_point_dim, *args, use_dyn_tree=False, carry_dim_seg1=None, coo_dim=6, **kwargs):
        super().__init__()
        if carry_dim_seg1 is None:
            carry_dim_seg1 = carry_dim
        self.segment1 = Segment(carry_dim_seg1, encoder_with_point_dim(3), *args, **kwargs)
        self.segment2 = Segment(carry_dim, encoder_with_point_dim(carry_dim_seg1), *args, **kwargs)

        self.encoder = self.segment2.encoder
        self.tree = self.segment1.encoder.tree
        self.num_parts = self.segment2.num_parts
        self.num_classes = self.segment2.num_classes
        self.inner_part_classfier = self.segment1.part_classfier
        self.part_classfier = self.segment2.part_classfier
        self.use_dyn_tree = use_dyn_tree
        self.coo_dim = coo_dim

    def forward(self, ans, inputs, *args, **kwargs):
        ans = ans.cuda()
        ans = self.segment1(ans, inputs, *args, **kwargs)
        self.inner_ans = ans
        if self.use_dyn_tree:
            coo, _, _ = torch.pca_lowrank(ans.detach(), q=self.coo_dim)
            inputs[0] = build_tree.dynamic_arrange(coo)

        ans = self.segment2(ans, inputs, *args, **kwargs)
        self.align_reg_loss = self.segment1.align_reg_loss
        return ans
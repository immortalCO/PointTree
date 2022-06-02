import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import build_tree
import random
from encoder import MLP

class MLPCounter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mlp = MLP(*args, **kwargs)

    def forward(self, ans):
        ans, counter = ans.split(ans.size(-1) - 1, dim=-1)
        ans = self.mlp(ans)
        ans = torch.cat([ans, counter], dim=-1)
        return ans


class Segment(nn.Module):
    def __init__(self, ancestor_dim, encoder, num_parts=0, num_classes=0, force_sample=False, part_cls_dropout=None):
        super().__init__()

        self.encoder = encoder
        self.layers = []
        self.tree = self.encoder.tree
        self.num_parts = num_parts
        self.num_classes = num_classes

        self.has_sample = force_sample
        for encoder_layer in self.encoder.layers:
            if encoder_layer.layer_type == 'sampled':
                self.has_sample = True

        MLPClass = MLPCounter if self.has_sample else MLP

        globe_dim = []

        for encoder_layer in reversed(self.encoder.layers):
            globe_dim.append(max(ancestor_dim, encoder_layer.feature_dim))

        for i, encoder_layer in enumerate(reversed(self.encoder.layers)):
            if i == 0:
                assert globe_dim[i] == encoder_layer.feature_dim
                input_dim = globe_dim[i] 
                push_down = nn.Identity()
            else:
                input_dim = globe_dim[i - 1] + encoder_layer.feature_dim
                push_down = MLPClass([input_dim, globe_dim[i]])


            push_down_sample = nn.Identity()
            if encoder_layer.layer_type == 'sampled':
                # push_down_sample = MLPClass([globe_dim[i], globe_dim[i + encoder.sample_layers - 1]])
                push_down_sample = MLPClass([input_dim, globe_dim[i + encoder.sample_layers - 1]])

            logging.info(f"layer {i} feature_dim = {encoder_layer.feature_dim} globe_dim = {globe_dim[i]}")

            self.layers.append(nn.ModuleList([push_down, push_down_sample]))

        self.layers = nn.ModuleList(self.layers)
        # self.cloud_classifier = MLP([self.encoder.dim, self.encoder.dim // 2, self.encoder.dim // 4, num_classes], last_bn=False)
        if num_parts > 0:
            self.part_classfier = MLP([ancestor_dim, ancestor_dim // 2, ancestor_dim // 4, num_parts], 
                last_bn=False, last_dropout=part_cls_dropout)
        else:
            self.part_classfier = torch.nn.Identity()

    def forward(self, *args, **kwargs):
        if self.has_sample:
            return self.forward_with_sample(*args, **kwargs)

        assert not self.has_sample
        features = self.encoder(*args, **kwargs)
        self.features = features
        batch_size = features.size(0)

        inputs = None

        for i, ((push_down, push_down_sample), (encoder_layer, features)) in enumerate(
            zip(self.layers, reversed(tuple(zip(self.encoder.layers, self.encoder.layer_output))))):

            if inputs is None:
                inputs = features
            else:
                inputs = torch.cat([features, inputs], dim=-1)

            ans = push_down(inputs)

            if encoder_layer.layer_type == 'leaf':
                break

            n = ans.size(1) * 2
            dim = ans.size(2)
            assert encoder_layer.child_lr.size(0) == n

            outputs = torch.empty(batch_size, n, dim, device='cuda')
            outputs.scatter_(1, encoder_layer.child_l[None, :, None].expand(batch_size, -1, dim), ans)
            outputs.scatter_(1, encoder_layer.child_r[None, :, None].expand(batch_size, -1, dim), ans)
            inputs = outputs


        def data_part(ans):
            return ans[:, :, :-1]

        def counter_part(ans):
            return ans[:, :, -1, None]

        node_features = torch.cat([ans, torch.ones(*ans.shape[:-1], 1, device='cuda')], dim=-1)
        dim = node_features.size(-1)
        arrange = self.encoder.arrange
        ans = torch.zeros(batch_size, arrange.max().item() + 1, dim, dtype=torch.float, device='cuda')
        ans.scatter_add_(1, arrange[:, :, None].cuda().expand(batch_size, -1, dim), node_features) 

        ans = data_part(ans) / counter_part(ans)
        self.align_reg_loss = self.encoder.align_reg_loss
        return ans


    def forward_with_sample(self, *args, **kwargs):

        features = self.encoder(*args, **kwargs)
        self.features = features
        batch_size = features.size(0)

        # if calc_cloud_logits:
        #     cloud_logits = self.cloud_classifier(features)

        scatter_list = [[] for _ in self.layers]
        scatter_list[0].append([torch.tensor([0], device='cuda'), torch.ones(batch_size, 1, 1, device='cuda')])


        def fetch(scatter_list):
            n = sum([val.size(1) for ind, val in scatter_list])
            dim = scatter_list[0][1].size(-1)
            ans = torch.empty(batch_size, n, dim, device='cuda')

            # print(f"fetch n = {n} dim = {dim}")

            for ind, val in scatter_list:
                # print(f"scatter item {val.shape}")
                assert val.size(-1) == dim
                ans.scatter_(1, ind[None, :, None].expand(batch_size, -1, dim), val)

            return ans


        for i, ((push_down, push_down_sample), (encoder_layer, features)) in enumerate(
            zip(self.layers, reversed(tuple(zip(self.encoder.layers, self.encoder.layer_output))))):

            inputs = fetch(scatter_list[i])
            # print(f"decoder fwd {i} {features.shape} {inputs.shape}")
            inputs = torch.cat([features, inputs], dim=-1)

            ans = push_down(inputs)
            # print(f"decoder push_down {ans.shape}")

            if encoder_layer.layer_type != 'leaf':
                scatter_list[i + 1].append([encoder_layer.child_l, ans])
                scatter_list[i + 1].append([encoder_layer.child_r, ans])

                if encoder_layer.layer_type == 'sampled':
                    # ans_sample = push_down_sample(ans)
                    ans_sample = push_down_sample(inputs)
                    # print(f"decoder push_down_sample {ans_sample.shape}")
                    ans_sample[:, :, -1] /= 2 ** self.encoder.sample_layers
                    scatter_list[i + self.encoder.sample_layers].append([encoder_layer.child_s, ans_sample])

        def data_part(ans):
            return ans[:, :, :-1]

        def counter_part(ans):
            return ans[:, :, -1, None]

        counter = counter_part(ans)
        node_features = torch.cat([data_part(ans) * counter, counter], dim=-1)

        dim = node_features.size(-1)
        arrange = self.encoder.arrange
        ans = torch.zeros(batch_size, arrange.max().item() + 1, dim, dtype=torch.float, device='cuda')
        ans.scatter_add_(1, arrange[:, :, None].cuda().expand(batch_size, -1, dim), node_features) 

        # print(f"scatter {ans.shape}")
        # print(f"counters = {ans[:, :, -1]}")

        ans = data_part(ans) / counter_part(ans)
        # ans = self.part_classfier(ans)

        # if calc_cloud_logits:
        #     ans = (ans, cloud_logits)

        return ans

    @staticmethod
    def convert_raw(raw_logits, cloud_logits, class_parts, cloud_logits_coef=1):

        # use static to make sure the model cannot see "class_parts"
        import torch.nn.functional as F

        cloud_logits = F.log_softmax(cloud_logits * cloud_logits_coef, dim=-1)
        logits = torch.empty_like(raw_logits).cuda()

        for c, cpart in enumerate(class_parts):
            logits[:, :, cpart] = cloud_logits[:, c, None, None] + F.log_softmax(raw_logits[:, :, cpart], dim=-1)

        return logits


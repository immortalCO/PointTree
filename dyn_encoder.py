import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import build_tree
import random
from copy import deepcopy

from encoder import Encoder, MLP, Alignment
from segment import Segment


class DynEncoder(torch.nn.Module):
	def __init__(self, model_size, OUTPUT='', dim=1024, inner_dim=64, coo_dim=6, num_layer=1, dim_layer0=32, dim_repeat_cut=0):
		super().__init__()

		point_dim = 3
		layers = []

		new_encoder = lambda point_dim : Encoder(model_size, 9999, dim, point_dim=point_dim, 
				OUTPUT=OUTPUT, dim_layer0=dim_layer0, dim_repeat_cut=dim_repeat_cut)

		for i in range(num_layer):
			encoder = new_encoder(point_dim)
			block = Segment(inner_dim, encoder)
			point_dim = inner_dim
			layers.append(block)

		self.layers = nn.ModuleList(layers)
		self.feed = new_encoder(point_dim)
		self.tree = self.feed.tree
		self.dim = self.feed.dim
		self.align_dim = self.feed.dim
		self.align_num = num_layer
		self.coo_dim = coo_dim
		self.cache = False

	def forward(self, ans, inputs, *args, **kwargs):
		ans = ans.cuda()
		self.align_feature = []
		self.cached_inputs = []
		for i, block in enumerate(self.layers):
			if i > 0:
				inputs[0] = build_tree.dynamic_arrange(coo)
			if self.cache:
				self.cached_inputs.append(deepcopy(inputs))
			out = block(ans, inputs, *args, **kwargs)
			self.align_feature.append(block.features)
			ans = out if i == 0 else (ans + out)
			coo, _, _ = torch.pca_lowrank(ans.detach(), q=self.coo_dim)

		inputs[0] = build_tree.dynamic_arrange(coo)
		if self.cache:
			self.cached_inputs.append(deepcopy(inputs))
		feature = self.feed(ans, inputs, *args, **kwargs)
		return feature


class SampleEncoder(torch.nn.Module):
	def __init__(self, encoder, sample_encoder, extra_sample_coef=1/8):
		super().__init__()

		encoder = encoder()
		sample_encoder = sample_encoder()
		self.encoder = Segment(sample_encoder.idim, encoder)
		self.sample_encoder = sample_encoder

		self.tree = encoder.tree
		self.dim = sample_encoder.dim
		self.align_dim = encoder.dim
		self.align_num = 1
		self.extra_sample_coef = extra_sample_coef
		self.debug = False


	def forward(self, pts, inputs, *args, **kwargs):
		pts = pts.cuda()
		pts = self.encoder(pts, inputs, *args, **kwargs)
		self.align_feature = [self.encoder.features]

		with torch.no_grad():
			batch = pts.shape[0]
			N = pts.shape[1]

			dep = len(self.sample_encoder.layers) - 1
			rnd = pts.detach().permute(2, 0, 1)
			mod = (dep - rnd.shape[0] % dep) % dep
			rnd = torch.cat([rnd, rnd[:mod]], dim=0)
			rnd = rnd.reshape(-1, dep, batch, N).sum(dim=0)
			
			ind = torch.arange(N, device='cuda')[None, None, :].expand(batch, 1, -1)

			for i, val in enumerate(rnd):
				batch, node, sub = ind.shape

				val = val.gather(1, ind.reshape(batch, -1)).reshape(batch, node, sub)
				lch = ind.gather(2, val.topk(int(sub * (0.5 + self.extra_sample_coef)), largest=False).indices)
				rch = ind.gather(2, val.topk(int(sub * (0.5 + self.extra_sample_coef)), largest=True).indices)

				ind = torch.cat([lch, rch], dim=1)

				if self.debug:
					print(f"i = {i} ind = {ind.shape}")


			if ind.shape[2] != 1:
				batch, node, sub = ind.shape
				cho = torch.randint(low=0, high=sub, size=[batch, node, 1], device='cuda')
				ind = ind.gather(2, cho)
			ind = ind.squeeze(2)

		if self.debug:
			print(f"i = # ind = {ind.shape}")

		inputs = [ind] + [inputs[-1]] * (len(self.sample_encoder.layers) - 1)
		feature = self.sample_encoder(pts, inputs, *args, **kwargs)
		return feature
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.network_util import initseq, RodriguesModule


class PoseRefinementModule(nn.Module):
	def __init__(self, module_cfg, **kwargs):
		super().__init__()

		self.cfg = module_cfg

		embedding_size = module_cfg.embedding_size
		block_mlps = [nn.Linear(embedding_size, module_cfg.mlp_width), nn.ReLU()]

		for _ in range(0, module_cfg.mlp_depth - 1):
			block_mlps += [nn.Linear(module_cfg.mlp_width, module_cfg.mlp_width), nn.ReLU()]

		self.refine_root = module_cfg.refine_root
		self.refine_t = module_cfg.refine_t

		self.total_bones = module_cfg.total_bones if self.refine_root else module_cfg.total_bones - 1
		block_mlps += [nn.Linear(module_cfg.mlp_width, 3 * self.total_bones)]

		self.block_mlps = nn.Sequential(*block_mlps)
		initseq(self.block_mlps)

		# init the weights of the last layer as very small value
		# -- at the beginning, we hope the rotation matrix can be identity
		init_val = 1e-5
		last_layer = self.block_mlps[-1]
		last_layer.weight.data.uniform_(-init_val, init_val)
		last_layer.bias.data.zero_()

		self.rodriguez = RodriguesModule()

	def forward(self, dst_posevec, **kwargs):
		rvec = self.block_mlps(dst_posevec).view(-1, 3)
		Rs = self.rodriguez(rvec).view(-1, self.total_bones, 3, 3)

		root_Rs = torch.eye(3, device=Rs.device, dtype=Rs.dtype)[None, None, :, :]\
			.repeat(Rs.shape[0], 1, 1, 1)
		Rs = torch.cat([root_Rs, Rs], dim=1)

		return Rs

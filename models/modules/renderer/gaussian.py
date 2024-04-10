import logging
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from utils.pc_util import cam_T_world
from utils.camera_util import focal2fov


class Renderer(nn.Module):
	def __init__(self, module_cfg, canonical_info, **kwargs):
		super().__init__()

		self.cfg = module_cfg
		self.renderer = GaussianRasterizer(None)

	def forward(self, xyzs_observation, appearance_feats, opacity, K, E, bg_col=None, skeleton_info=None, **kwargs):
		B, N, _ = xyzs_observation.shape
		assert B == 1

		assert 'cov' in skeleton_info
		cov = skeleton_info['cov']  # B x N x 3 x 3

		# set up the rasterization setting
		focalx, focaly = K[0, 0, 0].item(), K[0, 1, 1].item()
		px, py = K[0, 0, 2].item(), K[0, 1, 2].item()
		w, h = self.cfg.img_size
		fovx = focal2fov(focalx, w)
		fovy = focal2fov(focaly, h)
		tanfovx = math.tan(fovx * 0.5)
		tanfovy = math.tan(fovy * 0.5)

		znear = 0.001
		zfar = 100

		K_ndc = torch.tensor([
			[2 * focalx / w, 0, (2 * px - w) / w, 0],
			[0, 2 * focaly / h, (2 * py - h) / h, 0],
			[0, 0, zfar / (zfar - znear), -zfar * znear / (zfar - znear)],
			[0, 0, 1, 0]
		]).float().to(K.device)
		cam_center = E[0].T.inverse()[3, :3]

		feat = torch.cat([appearance_feats, torch.ones_like(opacity)], dim=-1)
		if bg_col is None:
			bg_col = feat.new_zeros([self.cfg.feat_dim])

		render_setting = GaussianRasterizationSettings(
			image_height=h,
			image_width=w,
			tanfovx=tanfovx,
			tanfovy=tanfovy,
			bg=bg_col,
			scale_modifier=1.,
			viewmatrix=E[0].T,
			projmatrix=E[0].T @ K_ndc.T,
			sh_degree=0,
			campos=cam_center,
			prefiltered=False,
			debug=False
		)
		self.renderer.raster_settings = render_setting

		means2D = torch.zeros_like(xyzs_observation[0].T, requires_grad=True)

		cov_packed = torch.stack([
			cov[0, :, 0, 0], cov[0, :, 0, 1], cov[0, :, 0, 2],
			cov[0, :, 1, 1], cov[0, :, 1, 2],
			cov[0, :, 2, 2]
		], dim=-1)

		C = feat.shape[-1]
		if C % 3 != 0:
			C_add = 3 - C % 3
			feat = torch.cat([feat, feat[..., :C_add]], dim=-1)
		preds = []
		for i in range(0, feat.shape[-1], 3):
			pred, _ = self.renderer(
				means3D=xyzs_observation[0].T,
				means2D=means2D,
				colors_precomp=feat[0, :, i:i+3],
				shs=None,
				opacities=opacity[0, :],
				scales=None,
				rotations=None,
				cov3D_precomp=cov_packed)
			preds.append(pred)
		pred = torch.cat(preds, dim=0)
		pred = pred[:C]

		# transpose the image
		# gaussian splatting defines x along height
		pred = pred.permute(1, 2, 0)[None]

		return pred[..., :-1], pred[..., -1]

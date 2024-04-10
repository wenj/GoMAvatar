import seaborn as sns
import numpy as np
import logging
import copy
import cv2
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.pose_refinement_module import PoseRefinementModule
from .modules.non_rigid_module import NonRigidModule
from .modules.appearance_module import AppearanceModule
from .modules.shadow_module import ShadowModule
from .modules.renderer import load_renderer

from utils.network_util import RodriguesModule
from utils.body_util import apply_lbs, get_global_RTs
from utils.pc_util import ndc_T_world, img_T_cam, cam_T_world, subdivide

from pytorch3d import ops
from pytorch3d.structures import Meshes
from pytorch3d.transforms.so3 import so3_exp_map, so3_log_map


def get_transformation_from_triangle_steiner(triangles, sigma=0.001):
	centroid = triangles.mean(dim=-2)

	f1 = 0.5 * (triangles[..., 2, :] - centroid)
	f2 = 1 / (2 * np.sqrt(3)) * (triangles[..., 1, :] - triangles[..., 0, :])
	t0 = torch.atan2((2 * f1 * f2).sum(-1), ((f1 * f1).sum(-1) - (f2 * f2).sum(-1))) / 2
	t0 = t0[..., None]

	axis0 = f1 * torch.cos(t0) + f2 * torch.sin(t0)
	axis1 = f1 * torch.cos(t0 + np.pi / 2) + f2 * torch.sin(t0 + np.pi / 2)

	normal = torch.cross(axis0, axis1, dim=-1)
	normal = F.normalize(normal, dim=-1) * sigma
	transform = torch.stack([axis0 * 2, axis1 * 2, normal], dim=-1)
	return transform


class Model(nn.Module):
	def __init__(self, model_cfg, canonical_info):
		super().__init__()

		self.cfg = model_cfg
		# override attributes
		module_list = ['appearance', 'canonical_geometry', 'renderer', 'non_rigid', 'normal_renderer']
		for module_name in module_list:
			module_cfg = getattr(model_cfg, module_name)
			module_cfg.img_size = model_cfg.img_size
		model_cfg.normal_renderer.eval_mode = model_cfg.eval_mode

		self.register_buffer('faces', torch.tensor(canonical_info['faces'].astype(int)))
		self.face_connectivity = self.get_face_connectivity(self.faces, torch.tensor(
			canonical_info['canonical_vertex'])).detach().cuda()
		target_edge_length = self.get_init_edge_length(self.faces,
													   torch.tensor(canonical_info['canonical_vertex'])).cuda()
		self.register_buffer('target_edge_length', target_edge_length)

		# lbs weights
		lbs_weights = torch.tensor(canonical_info['canonical_lbs_weights']).float().transpose(1, 0)
		lbs_weights_w_bg = torch.zeros_like(lbs_weights)
		lbs_weights_w_bg = torch.cat([lbs_weights_w_bg, torch.zeros_like(lbs_weights[:1])], dim=0)
		lbs_weights_w_bg[:-1] = lbs_weights
		if model_cfg.lbs_weights.refine:
			self.lbs_weights = nn.Parameter(torch.log(lbs_weights_w_bg))
		else:
			self.register_buffer('lbs_weights', lbs_weights_w_bg)

		# gaussian parameters
		self.vertices = nn.Parameter(torch.tensor(canonical_info['canonical_vertex']).float().transpose(1, 0))
		if model_cfg.canonical_geometry.deform_so3:
			self.so3 = nn.Parameter(torch.zeros([3, self.faces.shape[0]]).float())
		else:
			self.register_buffer('so3', torch.zeros([3, self.faces.shape[0]]).float())
		if model_cfg.canonical_geometry.deform_scale:
			self.scale = nn.Parameter(
				torch.ones([3, self.faces.shape[0]]).float() * model_cfg.canonical_geometry.radius_scale)
		else:
			self.register_buffer(
				'scale',
				torch.ones([3, self.faces.shape[0]]).float() * model_cfg.canonical_geometry.radius_scale)

		# colors
		self.appearance_module = AppearanceModule(model_cfg.appearance, canonical_info)

		# pose refinement
		if model_cfg.pose_refinement.name != 'none':
			self.pose_refinement_module = PoseRefinementModule(model_cfg.pose_refinement)
		else:
			self.pose_refinement_module = None

		# non-rigid motion
		if model_cfg.non_rigid.name != 'none':
			self.non_rigid_module = NonRigidModule(model_cfg.non_rigid)
		else:
			self.non_rigid_module = None

		# albedo renderer (gaussian splatting)
		self.renderer = load_renderer(model_cfg.renderer, canonical_info)

		# normal renderer (mesh rasterization) and shadow prediction
		if model_cfg.normal_renderer.name != 'none':
			self.normal_renderer = load_renderer(model_cfg.normal_renderer, canonical_info)
		else:
			self.normal_renderer = None
		if model_cfg.shadow_module.name != 'none':
			self.shadow_module = ShadowModule(model_cfg.shadow_module)
		else:
			self.shadow_module = None

	def get_face_connectivity(self, faces, vertices):
		mesh = Meshes(vertices[None], faces[None])
		faces_to_edges = mesh.faces_packed_to_edges_packed()
		max_edge_id = torch.max(faces_to_edges)
		connected_faces = []
		for i in range(max_edge_id):
			faces = torch.nonzero(faces_to_edges == i)[:, 0]
			if len(faces) > 1:
				connected_faces.append(faces)

		return torch.stack(connected_faces, dim=0)

	def get_init_edge_length(self, faces, vertices):
		mesh = Meshes(vertices[None], faces[None])
		edges_packed = mesh.edges_packed()  # (sum(E_n), 3)
		verts_packed = mesh.verts_packed()  # (sum(V_n), 3)
		verts_edges = verts_packed[edges_packed]
		v0, v1 = verts_edges.unbind(1)
		edge_length = (v0 - v1).norm(dim=1, p=2)
		return edge_length

	def subdivide(self, need_face_connectivity=True):
		vertices_canonical = self.vertices  # 3 x N
		NF = self.faces.shape[0]
		faces_centroid_canonical = vertices_canonical.permute(1, 0)[self.faces.reshape(-1)].reshape(NF, 3, -1).mean(1)
		appearance_feats, bg_feat = self.appearance_module(xyzs=faces_centroid_canonical.permute(1, 0))

		attributes = {
			'weights': self.lbs_weights.permute(1, 0).detach().cpu().numpy(),
		}
		xyzs_canonical, faces, attributes, edges, face_index = subdivide(
			vertices_canonical.permute(1, 0).detach().cpu().numpy(),
			self.faces.detach().cpu().numpy(),
			attributes,
			return_edges=True)

		self.vertices = nn.Parameter(torch.tensor(xyzs_canonical).permute(1, 0).to(self.vertices.device).float())
		self.faces = torch.tensor(faces).to(self.faces.device)
		if self.cfg.lbs_weights.refine:
			self.lbs_weights = nn.Parameter(torch.tensor(attributes['weights']).permute(1, 0).to(self.lbs_weights.device))
		else:
			self.lbs_weights = torch.tensor(attributes['weights']).permute(1, 0).to(self.lbs_weights.device)

		appearance_feats = appearance_feats[..., None].repeat(1, 1, 4).reshape(appearance_feats.shape[0], -1)
		self.appearance_module.set(appearance_feats)

		so3 = self.so3[..., None].repeat(1, 1, 4).reshape(self.so3.shape[0], -1)
		if self.cfg.canonical_geometry.deform_so3:
			self.so3 = nn.Parameter(so3)
		else:
			self.so3 = so3
		scale = self.scale[..., None].repeat(1, 1, 4).reshape(self.scale.shape[0], -1)
		if self.cfg.canonical_geometry.deform_scale:
			self.scale = nn.Parameter(scale)
		else:
			self.scale = scale

		self.target_edge_length = self.get_init_edge_length(
			self.faces,
			torch.tensor(xyzs_canonical).to(self.faces.device)
		)
		if need_face_connectivity:
			self.face_connectivity = self.get_face_connectivity(self.faces, self.vertices.permute(1, 0))
		else:
			self.face_connectivity = self.face_connectivity.new_zeros(self.target_edge_length.shape[0], 2)

	def print_info(self):
		logging.info(f'the number of effective points is {self.vertices.shape[1]}')

	def forward(self, K, E, cnl_gtfms, dst_Rs, dst_Ts, dst_posevec=None, canonical_joints=None,
				i_iter=1e7, # parameters used in non rigid module
				bgcolor=None,
				global_R=None, global_T=None, # parameters used for peoplesnapshot's test-time pose optimization
				tb=None):
		B = dst_Rs.shape[0]
		F = self.faces.shape[0]

		# pose refinement
		if self.pose_refinement_module is not None and i_iter >= self.cfg.pose_refinement.kick_in_iter:
			B, N_bones, _, _ = dst_Rs.shape
			delta_Rs = self.pose_refinement_module(dst_posevec)
			dst_Rs = torch.matmul(dst_Rs.reshape(B * N_bones, 3, 3), delta_Rs.reshape(B * N_bones, 3, 3)).reshape(B, N_bones, 3, 3)

		# compute vertices in canonical, pose and observation space
		vertices_canonical = self.vertices # 3 x N
		if self.non_rigid_module is not None and i_iter >= self.cfg.non_rigid.kick_in_iter:
			vertices_pose, _, _ = self.non_rigid_module(
				vertices_canonical.unsqueeze(0),
				dst_posevec,
				i_iter,
				R=None,
				S=None,
			)
			vertices_pose = vertices_pose[0] # 3 x N
		else:
			vertices_pose = vertices_canonical

		lbs_weights = self.lbs_weights
		vertices_observation = apply_lbs(
			vertices_pose.unsqueeze(0),
			*get_global_RTs(cnl_gtfms, dst_Rs, dst_Ts),
			lbs_weights)[0] # 3 x N

		if global_R is not None:
			# for people snapshot's test-pose optimization
			global_R = RodriguesModule()(global_R.unsqueeze(0))[0]
			vertices_observation = global_R @ vertices_observation + global_T[:, None]

		mesh_canonical = Meshes(vertices_canonical.permute(1, 0).unsqueeze(0), self.faces.unsqueeze(0))
		mesh_observation = Meshes(vertices_observation.permute(1, 0).unsqueeze(0), self.faces.unsqueeze(0))
		xyz_observation = vertices_observation.permute(1, 0)[self.faces.reshape(-1)].reshape(F, 3, -1).mean(dim=1)

		# get covariance matrix in observation space for rendering
		S = torch.diag_embed(self.scale.permute(1, 0))
		R = so3_exp_map(self.so3.permute(1, 0))
		cov_local = R @ S @ S.permute(0, 2, 1) @ R.permute(0, 2, 1)
		world_T_observation = get_transformation_from_triangle_steiner(
			vertices_observation.permute(1, 0)[self.faces.reshape(-1)].reshape(F, 3, -1),
			self.cfg.canonical_geometry.sigma) # F x 3 x 3
		cov_observation = world_T_observation @ cov_local @ world_T_observation.permute(0, 2, 1)

		# get color based on face centroid in canonical space
		F = self.faces.shape[0]
		faces_centroid_canonical = vertices_canonical.permute(1, 0)[self.faces.reshape(-1)].reshape(F, 3, -1).mean(1)
		appearance_feats, bg_feat = self.appearance_module(xyzs=faces_centroid_canonical.permute(1, 0))

		# render pseudo albedo
		opacity = appearance_feats.new_ones(B, F, 1)
		bg_col = torch.cat([bg_feat, bg_feat.new_zeros([1])])

		albedos, masks = self.renderer(
			xyz_observation.unsqueeze(0).permute(0, 2, 1),
			appearance_feats.unsqueeze(0).permute(0, 2, 1),
			opacity,
			K, E, bg_col=bg_col,
			skeleton_info={'cov': cov_observation.unsqueeze(0)})

		if tb is not None:
			pts_c_valid = vertices_canonical.permute(1, 0)
			pts_p_valid = vertices_pose.permute(1, 0)
			pts_o_valid = vertices_observation.permute(1, 0)
			opacity_valid = opacity[0].repeat(1, 3)
			rgbs_valid = appearance_feats.permute(1, 0)
			tb.summ_pointcloud('canonical/density', pts_c_valid.unsqueeze(0), None, faces=self.faces.unsqueeze(0))
			tb.summ_pointcloud('canonical/color', pts_c_valid.unsqueeze(0), rgbs_valid.unsqueeze(0), faces=self.faces.unsqueeze(0))
			tb.summ_pointcloud('pose/density', pts_p_valid.unsqueeze(0), None, faces=self.faces.unsqueeze(0))
			tb.summ_pointcloud('observation/density', pts_o_valid.unsqueeze(0), None, faces=self.faces.unsqueeze(0))
			tb.summ_pointcloud('observation/color', pts_o_valid.unsqueeze(0), rgbs_valid.unsqueeze(0), faces=self.faces.unsqueeze(0))

			lbs_weights_valid = self.lbs_weights[:-1].permute(1, 0)
			joint_colors = torch.tensor(np.array(sns.color_palette("tab10", lbs_weights_valid.shape[1]))).to(
				lbs_weights_valid.device).float()
			lbs_weights_colors = torch.matmul(lbs_weights_valid, joint_colors)
			tb.summ_pointcloud('canonical/lbs', pts_c_valid.unsqueeze(0), lbs_weights_colors.unsqueeze(0), faces=self.faces.unsqueeze(0))

		# compute the normals and render normal map
		normals = mesh_observation.verts_normals_padded()
		R = E[:, :3, :3]
		normals = torch.bmm(R, normals.permute(0, 2, 1)).permute(0, 2, 1)
		normal, normal_mask = self.normal_renderer(vertices_observation.unsqueeze(0), normals, K, E, faces=self.faces)

		if tb is not None:
			tb.summ_image('model/normal', (1 - (normal + 1) * 0.5)[0].permute(2, 0, 1))
			tb.summ_image('model/normal_mask', normal_mask[:1, ..., 0])

		# predict pseudo shading
		B, H, W, _ = normal.shape
		shadings = self.shadow_module(normal.reshape(-1, H * W, 3))  # B x N x 1
		shadings = shadings.reshape(B, H, W, 1) * 2
		if tb is not None:
			tb.summ_image('model/shadow', shadings[0].permute(2, 0, 1) / torch.max(shadings[0]))

		rgbs = albedos * shadings

		outputs = {}
		if self.training:
			# auxiliary outputs for loss
			outputs['colors'] = appearance_feats.permute(1, 0)
			outputs['face_connectivity'] = self.face_connectivity
			outputs['mesh'] = mesh_observation
			outputs['mesh_canonical'] = mesh_canonical
			outputs['target_edge_length'] = self.target_edge_length

			outputs['albedo'] = albedos[0]
			outputs['normal'] = normal
			outputs['normal_mask'] = normal_mask[..., 0]
			outputs['shadow'] = shadings

		return rgbs, masks, outputs

	def get_param_groups(self, cfg):
		param_groups = [
			{'name': 'lbs_weights', 'params': self.lbs_weights, 'lr': cfg.lr.lbs_weights},
			{'name': 'appearance', 'params': self.appearance_module.parameters(), 'lr': cfg.lr.appearance},
		]

		param_groups.append({'name': 'canonical_geometry_xyz', 'params': self.vertices, 'lr': cfg.lr.canonical_geometry_xyz})
		param_groups.append({'name': 'canonical_geometry', 'params': self.scale, 'lr': cfg.lr.canonical_geometry})
		param_groups.append({'name': 'canonical_geometry', 'params': self.so3, 'lr': cfg.lr.canonical_geometry})

		if self.non_rigid_module is not None:
			param_groups.append(
				{'name': 'non_rigid', 'params': self.non_rigid_module.parameters(), 'lr': cfg.lr.non_rigid})
		if self.pose_refinement_module is not None:
			param_groups.append({'name': 'pose_refinement', 'params': self.pose_refinement_module.parameters(),
								 'lr': cfg.lr.pose_refinement})
		if self.shadow_module is not None:
			param_groups.append({'name': 'shadow', 'params': self.shadow_module.parameters(), 'lr': cfg.lr.shadow})
		print(param_groups)
		return param_groups

	def export_canonical_pointcloud(self):
		xyzs_canonical = self.vertices.permute(1, 0)

		NF = self.faces.shape[0]
		faces_centroid_canonical = xyzs_canonical[self.faces.reshape(-1)].reshape(NF, 3, -1).mean(1)
		appearance_feats, bg_feat = self.appearance_module(xyzs=faces_centroid_canonical.permute(1, 0))
		rgbs = appearance_feats.permute(1, 0)

		lbs_weights = self.lbs_weights
		opacity = 1 - lbs_weights[-1]  # N

		S = torch.diag_embed(self.scale.permute(1, 0))
		R = so3_exp_map(self.so3.permute(1, 0))
		cov_local = R @ S @ S.permute(0, 2, 1) @ R.permute(0, 2, 1)
		world_T_observation = get_transformation_from_triangle_steiner(
			xyzs_canonical[self.faces.reshape(-1)].reshape(NF, 3, -1))  # F x 3 x 3
		cov_observation = world_T_observation @ cov_local @ world_T_observation.permute(0, 2, 1)

		print(xyzs_canonical.shape, opacity.shape, rgbs.shape, cov_observation.shape)

		return xyzs_canonical, None, opacity, rgbs, {'cov': cov_observation}

	def export_warped_pointcloud(self, cnl_gtfms, dst_Rs, dst_Ts, dst_posevec=None, i_iter=1e7):
		if self.pose_refinement_module is not None and i_iter >= self.cfg.pose_refinement.kick_in_iter:
			B, N_bones, _, _ = dst_Rs.shape
			delta_Rs, delta_Ts = self.pose_refinement_module(dst_posevec)
			dst_Rs = torch.matmul(dst_Rs.reshape(B * N_bones, 3, 3), delta_Rs.reshape(B * N_bones, 3, 3)).reshape(B, N_bones, 3, 3)
			dst_Ts = dst_Ts + delta_Ts

		vertices_canonical = self.vertices  # 3 x N
		if self.non_rigid_module is not None and i_iter >= self.cfg.non_rigid.kick_in_iter:
			vertices_pose, _, _ = self.non_rigid_module(
				vertices_canonical.unsqueeze(0),
				dst_posevec,
				i_iter,
				R=None,
				S=None,
			)
			vertices_pose = vertices_pose[0]  # 3 x N
		else:
			vertices_pose = vertices_canonical

		vertices_observation = apply_lbs(
			vertices_pose.unsqueeze(0),
			*get_global_RTs(cnl_gtfms, dst_Rs, dst_Ts, use_smplx=self.cfg.use_smplx),
			self.lbs_weights)[0]  # 3 x N

		NF = self.faces.shape[0]
		xyz_observation = vertices_observation.permute(1, 0)[self.faces.reshape(-1)].reshape(NF, 3, -1).mean(dim=1)

		# get covariance matrix in observation space for rendering
		S = torch.diag_embed(self.scale.permute(1, 0))
		R = so3_exp_map(self.so3.permute(1, 0))
		cov_local = R @ S @ S.permute(0, 2, 1) @ R.permute(0, 2, 1)
		world_T_observation = get_transformation_from_triangle_steiner(
			vertices_observation.permute(1, 0)[self.faces.reshape(-1)].reshape(NF, 3, -1))  # F x 3 x 3
		cov_observation = world_T_observation @ cov_local @ world_T_observation.permute(0, 2, 1)

		# get color based on face centroid in canonical space
		faces_centroid_canonical = vertices_canonical.permute(1, 0)[self.faces.reshape(-1)].reshape(NF, 3, -1).mean(1)
		appearance_feats, bg_feat = self.appearance_module(xyzs=faces_centroid_canonical.permute(1, 0))

		lbs_weights = self.lbs_weights
		opacity = 1 - lbs_weights[-1]  # N

		return vertices_observation.permute(1, 0), vertices_observation.permute(1, 0), opacity, appearance_feats, {'cov': cov_observation}

	def get_lbs_weights(self):
		return self.lbs_weights

import os
import pickle
import logging

import numpy as np
import cv2
import torch
import torch.utils.data

from utils.image_util import load_image
from utils.body_util import \
	body_pose_to_body_RTs, \
	get_canonical_global_tfms, \
	approx_gaussian_bone_volumes, \
	get_joints_from_pose
from utils.file_util import list_files, split_path
from utils.camera_util import \
	apply_global_tfm_to_camera, \
	get_rays_from_KRT, \
	rays_intersect_3d_bbox, \
	rotate_camera_by_frame_idx


class Dataset(torch.utils.data.Dataset):
	ROT_CAM_PARAMS = {
		'zju_mocap': {'rotate_axis': 'z', 'inv_angle': True},
		'wild': {'rotate_axis': 'y', 'inv_angle': False}
	}

	def __init__(
			self,
			dataset_path,
			frame_idx=0,
			total_frames=100,
			keyfilter=None,
			bgcolor=None,
			src_type="zju_mocap",
			target_size=None,
			**_):

		self.cfg = {
			'bbox_offset': 0.3,
			'resize_img_scale': [0.5, 0.5],
		}

		logging.info(f'[Dataset Path] {dataset_path}')

		self.dataset_path = dataset_path
		self.image_dir = os.path.join(dataset_path, 'images')

		self.canonical_joints, self.canonical_bbox, self.canonical_vertex, self.canonical_lbs_weights, self.edges, self.faces = \
			self.load_canonical_joints()

		cameras = self.load_train_cameras()
		mesh_infos = self.load_train_mesh_infos()

		framelist = self.load_train_frames()

		self.train_frame_idx = frame_idx
		logging.info(f' -- Frame Idx: {self.train_frame_idx}')

		self.total_frames = total_frames
		logging.info(f' -- Total Rendered Frames: {self.total_frames}')

		self.train_frame_name = framelist[self.train_frame_idx]
		self.train_camera = cameras[framelist[self.train_frame_idx]]
		self.train_mesh_info = mesh_infos[framelist[self.train_frame_idx]]

		self.bgcolor = bgcolor if bgcolor is not None else [255., 255., 255.]
		self.keyfilter = keyfilter
		self.src_type = src_type

		if target_size is not None:
			self.cfg['target_size'] = target_size

	def load_canonical_joints(self):
		cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
		with open(cl_joint_path, 'rb') as f:
			cl_joint_data = pickle.load(f)
		canonical_joints = cl_joint_data['joints'].astype('float32')
		canonical_bbox = self.skeleton_to_bbox(canonical_joints)

		canonical_vertex = cl_joint_data['vertex'].astype('float32')
		canonical_lbs_weights = cl_joint_data['weights'].astype('float32')

		if 'edges' in cl_joint_data:
			canonical_edges = cl_joint_data['edges'].astype(int)
		else:
			canonical_edges = None

		if 'faces' in cl_joint_data:
			canonical_faces = cl_joint_data['faces']
		else:
			canonical_faces = None

		return canonical_joints, canonical_bbox, canonical_vertex, canonical_lbs_weights, canonical_edges, canonical_faces

	def load_train_cameras(self):
		cameras = None
		with open(os.path.join(self.dataset_path, 'cameras.pkl'), 'rb') as f:
			cameras = pickle.load(f)
		return cameras

	def skeleton_to_bbox(self, skeleton):
		min_xyz = np.min(skeleton, axis=0) - self.cfg['bbox_offset']
		max_xyz = np.max(skeleton, axis=0) + self.cfg['bbox_offset']

		return {
			'min_xyz': min_xyz,
			'max_xyz': max_xyz
		}

	def load_train_mesh_infos(self):
		mesh_infos = None
		with open(os.path.join(self.dataset_path, 'mesh_infos.pkl'), 'rb') as f:
			mesh_infos = pickle.load(f)

		for frame_name in mesh_infos.keys():
			bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
			mesh_infos[frame_name]['bbox'] = bbox

		return mesh_infos

	def load_train_frames(self):
		img_paths = list_files(os.path.join(self.dataset_path, 'images'),
							   exts=['.png'])
		return [split_path(ipath)[1] for ipath in img_paths]

	def query_dst_skeleton(self):
		return {
			'poses': self.train_mesh_info['poses'].astype('float32'),
			'dst_tpose_joints': \
				self.train_mesh_info['tpose_joints'].astype('float32'),
			'bbox': self.train_mesh_info['bbox'].copy(),
			'Rh': self.train_mesh_info['Rh'].astype('float32'),
			'Th': self.train_mesh_info['Th'].astype('float32')
		}

	def get_freeview_camera(self, E, frame_idx, total_frames, trans=None):
		E = rotate_camera_by_frame_idx(
			extrinsics=E,
			frame_idx=frame_idx,
			period=total_frames,
			trans=trans,
			**self.ROT_CAM_PARAMS[self.src_type])
		K = self.train_camera['intrinsics'].copy()
		return K, E

	def load_image(self, frame_name, bg_color):
		imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
		orig_img = np.array(load_image(imagepath))
		orig_H, orig_W, _ = orig_img.shape

		maskpath = os.path.join(self.dataset_path,
								'masks',
								'{}.png'.format(frame_name))
		alpha_mask = np.array(load_image(maskpath))

		if 'distortions' in self.train_camera:
			K = self.train_camera['intrinsics']
			D = self.train_camera['distortions']
			orig_img = cv2.undistort(orig_img, K, D)
			alpha_mask = cv2.undistort(alpha_mask, K, D)

		alpha_mask = alpha_mask / 255.
		img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
		if 'target_size' in self.cfg:
			w, h = self.cfg['target_size']
			img = cv2.resize(img, [w, h],
							 interpolation=cv2.INTER_LANCZOS4)
			alpha_mask = cv2.resize(alpha_mask, [w, h],
									interpolation=cv2.INTER_LINEAR)
		else:
			if self.cfg['resize_img_scale'] != 1.:
				img = cv2.resize(img, None,
								 fx=self.cfg['resize_img_scale'][0],
								 fy=self.cfg['resize_img_scale'][1],
								 interpolation=cv2.INTER_LANCZOS4)
				alpha_mask = cv2.resize(alpha_mask, None,
										fx=self.cfg['resize_img_scale'][0],
										fy=self.cfg['resize_img_scale'][1],
										interpolation=cv2.INTER_LINEAR)

		return img, alpha_mask, orig_W, orig_H

	def __len__(self):
		return self.total_frames

	def __getitem__(self, idx):
		frame_name = self.train_frame_name
		results = {
			'frame_name': self.train_frame_name + f'_v{idx:04d}'
		}

		bgcolor = np.array(self.bgcolor, dtype='float32')

		img, alpha, orig_W, orig_H = self.load_image(frame_name, bgcolor)
		img = (img / 255.).astype('float32')
		H, W = img.shape[0:2]

		dst_skel_info = self.query_dst_skeleton()
		dst_bbox = dst_skel_info['bbox']
		dst_poses = dst_skel_info['poses'].reshape(-1)
		dst_tpose_joints = dst_skel_info['dst_tpose_joints']
		dst_Rh = dst_skel_info['Rh']
		dst_Th = dst_skel_info['Th']

		E = self.train_camera['extrinsics']
		K, E = self.get_freeview_camera(
			E=E,
			frame_idx=idx,
			total_frames=self.total_frames,
			trans=dst_Th)
		if 'target_size' in self.cfg:
			scale_w, scale_h = self.cfg['target_size'][0] / orig_W, self.cfg['target_size'][1] / orig_H
		else:
			scale_w, scale_h = self.cfg['resize_img_scale']
		K[:1] *= scale_w
		K[1:2] *= scale_h

		E = apply_global_tfm_to_camera(
			E=E,
			Rh=dst_Rh,
			Th=dst_Th)

		results.update({
			'K': K.astype(np.float32),
			'E': E.astype(np.float32),
		})

		results['target_rgbs'] = img

		dst_Rs, dst_Ts = body_pose_to_body_RTs(dst_poses, dst_tpose_joints)
		cnl_gtfms = get_canonical_global_tfms(self.canonical_joints)
		results.update({
			'dst_Rs': dst_Rs,
			'dst_Ts': dst_Ts,
			'cnl_gtfms': cnl_gtfms
		})

		# 1. ignore global orientation
		# 2. add a small value to avoid all zeros
		dst_posevec_69 = dst_poses[3:] + 1e-2
		results.update({
			'dst_posevec': dst_posevec_69,
		})

		return results

	def get_canonical_info(self):
		info = {
			'canonical_joints': self.canonical_joints,
			'canonical_bbox': {
				'min_xyz': self.canonical_bbox['min_xyz'],
				'max_xyz': self.canonical_bbox['max_xyz'],
				'scale_xyz': self.canonical_bbox['max_xyz'] - self.canonical_bbox['min_xyz'],
			},
			'canonical_vertex': self.canonical_vertex,
			'canonical_lbs_weights': self.canonical_lbs_weights,
			'edges': self.edges,
			'faces': self.faces,
		}
		return info

	def get_all_Es(self):
		Es = []
		for idx in range(self.total_frames):
			dst_skel_info = self.query_dst_skeleton()
			dst_Rh = dst_skel_info['Rh']
			dst_Th = dst_skel_info['Th']

			E = self.train_camera['extrinsics']
			K, E = self.get_freeview_camera(
				E=E,
				frame_idx=idx,
				total_frames=self.total_frames,
				trans=dst_Th)
			E = apply_global_tfm_to_camera(
				E=E,
				Rh=dst_Rh,
				Th=dst_Th)

			Es.append(E)
		return np.stack(Es, axis=0)
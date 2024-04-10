import os
import pickle
import numpy as np
import cv2
import logging
import pickle

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
    get_camrot

cfg = {
    'bbox_offset': 0.3,
    'resize_img_scale': 2.0,
}

KEYFILTER = ['rays', 'motion_bases', 'motion_weights_priors', 'cnl_bbox', 'dst_posevec_69']


class Dataset(torch.utils.data.Dataset):
    RENDER_SIZE = 512
    CAM_PARAMS = {
        'radius': 8.0, 'focal': 1250.
    }

    def __init__(
            self,
            dataset_path,
            pose_path,
            keyfilter=KEYFILTER,
            bgcolor=[0.0, 0.0, 0.0],
            debug=False,
            src_type='wild',
            **_):

        self.debug = debug
        self.src_type = src_type

        logging.info(f'[Dataset Path] {dataset_path}')
        logging.info(f'[Pose Path] {pose_path}')

        self.pose_path = pose_path

        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, 'images')
        self.total_frames = len(os.listdir(self.image_dir))

        self.canonical_joints, self.canonical_bbox, self.canonical_vertex, self.canonical_lbs_weights, self.edges, self.faces = \
            self.load_canonical_joints()

        self.mesh_infos = self.load_train_mesh_infos()
        self.pose_infos = self.load_mdm_pose_infos(self.pose_path)
        self.total_frames = len(self.pose_infos['Rh'])

        # setup the camera
        K, E = self.setup_camera(img_size=self.RENDER_SIZE,
                                 **self.CAM_PARAMS)
        self.camera = {
            'K': [K] * self.total_frames,
            'E': [E] * self.total_frames
        }

        self.keyfilter = keyfilter
        self.bgcolor = bgcolor

        # get the bounding box of canonical volume
        if 'cnl_bbox' in self.keyfilter:
            self.cnl_bbox_min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
            self.cnl_bbox_max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
            self.cnl_bbox_scale_xyz = 2.0 / (self.cnl_bbox_max_xyz - self.cnl_bbox_min_xyz)
            assert np.all(self.cnl_bbox_scale_xyz >= 0)

    @staticmethod
    def setup_camera(img_size, radius, focal):
        x = 0.
        y = 1.2
        z = radius
        campos = np.array([x, y, z], dtype='float32')
        camrot = get_camrot(campos,
                            lookat=np.array([0, y, 0.]),
                            inv_camera=True)

        E = np.eye(4, dtype='float32')
        E[:3, :3] = camrot
        E[:3, 3] = -camrot.dot(campos)

        K = np.eye(3, dtype='float32')
        K[0, 0] = focal
        K[1, 1] = focal
        K[:2, 2] = img_size / 2.

        return K, E

    def load_canonical_joints(self, smpl_upscale_factor=1):
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

        if smpl_upscale_factor > 1:
            assert 'faces' in cl_joint_data
            vertices = canonical_vertex
            faces = cl_joint_data['faces']
            attributes = {
                'weights': canonical_lbs_weights
            }
            for _ in range(smpl_upscale_factor - 1):
                vertices, faces, attributes, edges = subdivide(vertices, faces, attributes, return_edges=True)
            canonical_vertex = vertices
            canonical_lbs_weights = attributes['weights']
            canonical_edges = edges
            canonical_faces = faces

        return canonical_joints, canonical_bbox, canonical_vertex, canonical_lbs_weights, canonical_edges, canonical_faces

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg['bbox_offset']
        max_xyz = np.max(skeleton, axis=0) + cfg['bbox_offset']

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def load_mdm_pose_infos(self, path):
        data = dict(np.load(path, allow_pickle=True).item())
        poses = np.transpose(data['thetas_ori'].cpu().numpy(), (2, 0, 1))
        Rh = poses[:, 0].copy()
        Th = np.transpose(data['root_translation'], (1, 0))
        poses[:, 0] = 0.
        poses = poses.reshape(poses.shape[0], -1)
        pose_infos = {
            'poses': poses,
            'Rh': Rh,
            'Th': Th,
        }
        return pose_infos

    def load_train_mesh_infos(self, path=None):
        if path == None:
            path = self.dataset_path
        mesh_infos = None
        with open(os.path.join(path, 'mesh_infos.pkl'), 'rb') as f:
            mesh_infos = pickle.load(f)

        for frame_name in mesh_infos.keys():
            bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
            mesh_infos[frame_name]['bbox'] = bbox

        return mesh_infos

    def query_dst_skeleton(self, idx):
        frame_name = 'frame_000000'
        return {
            'poses': self.pose_infos['poses'][idx].astype('float32'),
            'dst_tpose_joints': \
                self.canonical_joints,
            'bbox': self.skeleton_to_bbox(get_joints_from_pose(
                self.pose_infos['poses'][idx],
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'))),
            'Rh': self.pose_infos['Rh'][idx].astype('float32'),
            'Th': self.pose_infos['Th'][0].astype('float32')
        }

    def load_image(self, frame_name, bg_color):
        imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
        orig_img = np.array(load_image(imagepath))

        maskpath = os.path.join(self.dataset_path,
                                'masks',
                                '{}.png'.format(frame_name))
        alpha_mask = np.array(load_image(maskpath))

        # undistort image
        if 'distortions' in self.camera:
            K = self.camera['intrinsics']
            D = self.camera['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        alpha_mask = alpha_mask / 255.
        img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
        if cfg['resize_img_scale'] != 1.:
            img = cv2.resize(img, None,
                             fx=cfg['resize_img_scale'],
                             fy=cfg['resize_img_scale'],
                             interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None,
                                    fx=cfg['resize_img_scale'],
                                    fy=cfg['resize_img_scale'],
                                    interpolation=cv2.INTER_LINEAR)

        return img, alpha_mask

    def get_total_frames(self):
        return len(self.past_framelist)

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        if self.src_type == 'zju-mocap':
            frame_name = 'frame_{:06d}'.format(idx)
        else:
            frame_name = '{:06d}'.format(idx)
        frame_name = 'frame_{:06d}'.format(idx)
        results = {
            'frame_name': frame_name
        }

        if self.bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')

        img, alpha = self.load_image(frame_name, bgcolor)
        img = (img / 255.).astype('float32')

        H, W = img.shape[0:2]
        H, W = 512, 512

        dst_skel_info = self.query_dst_skeleton(idx)
        dst_poses = dst_skel_info['poses']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']

        # dst_poses[19*3] -= 0.6

        results.update({
            'dst_poses': dst_poses,
            'dst_tpose_joints': dst_tpose_joints,
        })

        K = self.camera['K'][idx].copy()

        E = self.camera['E'][idx]

        # recover
        E = apply_global_tfm_to_camera(
            E=E,
            Rh=dst_skel_info['Rh'],
            Th=dst_skel_info['Th'] - self.canonical_joints[0])

        results.update({
            'K': K.astype(np.float32),
            'E': E.astype(np.float32),
        })

        # fake rgbs and masks
        results['target_rgbs'] = np.zeros([H, W, 3], dtype=np.float32)
        results['target_masks'] = np.zeros([H, W], dtype=np.float32)

        dst_Rs, dst_Ts = body_pose_to_body_RTs(
            dst_poses, dst_tpose_joints
        )
        cnl_gtfms = get_canonical_global_tfms(
            self.canonical_joints)
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
            'joints': get_joints_from_pose(dst_poses, dst_tpose_joints),
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
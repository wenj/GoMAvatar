import os
import pickle
import numpy as np
import cv2
import logging

import torch
import torch.utils.data

from utils.image_util import load_image
from utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from utils.file_util import list_files, split_path
from utils.camera_util import \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox

cfg = {
    'bbox_offset': 0.3,
    'resize_img_scale': 0.5,
}


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            raw_dataset_path,
            dataset_path,
            test_type='view',
            bgcolor=None,
            exclude_training_view=True,
            exclude_view=0,
            skip=30,
            **_):

        logging.info(f'[Raw dataset path]: {raw_dataset_path}')
        logging.info(f'[Dataset Path]: {dataset_path}')
        if exclude_training_view:
            logging.info(f'[Exclude view]: {exclude_view}')

        self.raw_dataset_path = raw_dataset_path
        self.dataset_path = dataset_path

        self.exclude_training_view = exclude_training_view
        self.exclude_view = exclude_view

        self.canonical_joints, self.canonical_bbox, self.canonical_vertex, self.canonical_lbs_weights, self.edges, self.faces = \
            self.load_canonical_joints()

        self.cameras = self.load_test_cameras()
        self.mesh_infos = self.load_train_mesh_infos()

        self.framelist = self.load_train_frames()
        if test_type == 'view':
            logging.info('use monohuman split - testing novel view')
            self.framelist = self.framelist[:-(len(self.framelist) // 5)]
        elif test_type == 'pose':
            logging.info('use monohuman split - testing novel pose')
            self.framelist = self.framelist[-(len(self.framelist) // 5):]
        else:
            raise NotImplementedError(f'unknown test_type {test_type}')
        self.framelist = self.framelist[::skip]
        logging.info(f' -- Total Frames: {self.get_total_frames() * len(self.cameras)}')

        self.bgcolor = bgcolor

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

    def load_test_cameras(self):
        anno_path = os.path.join(self.raw_dataset_path, 'annots.npy')
        annots = np.load(anno_path, allow_pickle=True).item()

        # load cameras
        cameras = {}
        cams = annots['cams']
        for view_id in range(len(cams['K'])):
            if self.exclude_training_view and view_id == self.exclude_view:
                continue

            cam_Ks = np.array(cams['K'])[view_id].astype('float32')
            cam_Rs = np.array(cams['R'])[view_id].astype('float32')
            cam_Ts = np.array(cams['T'])[view_id].astype('float32') / 1000.
            cam_Ds = np.array(cams['D'])[view_id].astype('float32')

            K = cam_Ks  # (3, 3)
            D = cam_Ds[:, 0]
            E = np.eye(4)  # (4, 4)
            cam_T = cam_Ts[:3, 0]
            E[:3, :3] = cam_Rs
            E[:3, 3] = cam_T

            cameras[view_id] = {
                'intrinsics': K,
                'extrinsics': E,
                'distortions': D
            }
        return cameras

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg['bbox_offset']
        max_xyz = np.max(skeleton, axis=0) + cfg['bbox_offset']

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def load_train_mesh_infos(self):
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
    
    def query_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            'dst_tpose_joints': \
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            'bbox': self.mesh_infos[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32')
        }

    def load_mask(self, img_name):
        msk_path = os.path.join(self.raw_dataset_path, 'mask',
                                img_name)[:-4] + '.png'
        msk = np.array(load_image(msk_path))[:, :, 0]
        msk = (msk != 0).astype(np.uint8)

        msk_path = os.path.join(self.raw_dataset_path, 'mask_cihp',
                                img_name)[:-4] + '.png'
        msk_cihp = np.array(load_image(msk_path))[:, :, 0]
        msk_cihp = (msk_cihp != 0).astype(np.uint8)

        msk = (msk | msk_cihp).astype(np.uint8)
        msk[msk == 1] = 255

        return msk
    
    def load_image(self, view_id, frame_name, bg_color):
        imagepath = os.path.join(self.raw_dataset_path, 'Camera_B{}'.format(view_id + 1), '{:06d}.jpg'.format(frame_name))
        orig_img = np.array(load_image(imagepath))

        alpha_mask = self.load_mask(os.path.join('Camera_B{}'.format(view_id + 1), '{:06d}.png'.format(frame_name)))
        
        # undistort image
        if view_id in self.cameras and 'distortions' in self.cameras[view_id]:
            K = self.cameras[view_id]['intrinsics']
            D = self.cameras[view_id]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        alpha_mask = alpha_mask / 255.
        alpha_mask = alpha_mask[:, :, None]
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
        return len(self.framelist)

    def __len__(self):
        return self.get_total_frames() * len(self.cameras)

    def __getitem__(self, idx):
        view_id = sorted(self.cameras.keys())[idx % len(self.cameras)]
        frame_name = self.framelist[idx // len(self.cameras)]
        frame_id = int(frame_name.split('_')[1])

        results = {
            'frame_name': 'Camera_B{}_{}'.format(view_id + 1, frame_name)
        }

        if self.bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')

        img, alpha = self.load_image(view_id, frame_id, bgcolor)
        img = (img / 255.).astype('float32')

        H, W = img.shape[0:2]

        dst_skel_info = self.query_dst_skeleton(frame_name)
        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']

        assert view_id in self.cameras
        K = self.cameras[view_id]['intrinsics'][:3, :3].copy()
        K[:2] *= cfg['resize_img_scale']

        E = self.cameras[view_id]['extrinsics']
        E = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'])
        R = E[:3, :3]
        T = E[:3, 3]
        results.update({
            'K': K.astype(np.float32),
            'E': E.astype(np.float32),
        })

        results['target_rgbs'] = img
        results['target_masks'] = alpha.astype(np.float32)

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

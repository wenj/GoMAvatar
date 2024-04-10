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
    get_joints_from_pose
from utils.file_util import list_files, split_path
from utils.camera_util import apply_global_tfm_to_camera


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataset_path,
            keyfilter=None,
            maxframes=-1,
            bgcolor=None,
            skip=1,
            target_size=None,
            crop_size=[-1, -1],
            prefetch=False,
            split_for_pose=False,
    ):
        self.cfg = {
            'bbox_offset': 0.3,
            'resize_img_scale': [0.5, 0.5],
        }

        logging.info(f'[Dataset Path]: {dataset_path}')

        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, 'images')

        self.canonical_joints, self.canonical_bbox, self.canonical_vertex, self.canonical_lbs_weights, self.edges, self.faces = \
            self.load_canonical_joints()

        self.cameras = self.load_train_cameras()
        self.mesh_infos = self.load_train_mesh_infos()

        self.framelist = self.load_train_frames()
        self.framelist = self.framelist[::skip]
        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]
        if split_for_pose:
            logging.info('use monohuman split')
            self.framelist = self.framelist[:-(len(self.framelist) // 5)]
        logging.info(f' -- Total Frames: {self.get_total_frames()}')

        self.keyfilter = keyfilter
        self.bgcolor = bgcolor

        if target_size is not None:
            self.cfg['target_size'] = target_size
        self.cfg['crop_size'] = crop_size

        self.prefetch = prefetch
        if prefetch:
            self.preload = {}
            self.normals = {}
            for idx in range(self.get_total_frames()):
                frame_name = self.framelist[idx]

                bgcolor = np.zeros([3]).astype('float32')
                img, alpha, orig_W, orig_H = self.load_image(frame_name, bgcolor)
                self.preload[frame_name] = [img, alpha, orig_W, orig_H]

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
    
    def query_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            'dst_tpose_joints': \
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            'bbox': self.mesh_infos[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32')
        }
    
    def load_image(self, frame_name, bg_color):
        imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
        orig_img = np.array(load_image(imagepath))
        orig_H, orig_W, _ = orig_img.shape

        maskpath = os.path.join(self.dataset_path,
                                'masks',
                                '{}.png'.format(frame_name))
        alpha_mask = np.array(load_image(maskpath))
        
        # undistort image
        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
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

    def crop_image(self, img, mask, K):
        crop_w, crop_h = self.cfg['crop_size']
        h, w, _ = img.shape

        h_center, w_center, _ = np.stack(np.nonzero(mask), axis=-1).mean(axis=0).astype(int)
        if h_center + (crop_h + 1) // 2 > h:
            h_center = h - (crop_h + 1) // 2
        if h_center - crop_h // 2 < 0:
            h_center = crop_h // 2
        if w_center + (crop_w + 1) // 2 > w:
            w_center = w - (crop_w + 1) // 2
        if w_center - crop_w // 2 < 0:
            w_center = crop_w // 2
        h_left = h_center - crop_h // 2
        w_left = w_center - crop_w // 2

        while True:
            rand_w, rand_h = np.random.randint(max(0, w_left - 50), min(w_left + 50, w - crop_w + 1)), np.random.randint(max(0, h_left - 50), min(h_left + 50, h - crop_h + 1))
            crop_mask = mask[rand_h:rand_h + crop_h, rand_w:rand_w + crop_w]
            if np.sum(crop_mask) < 20:
                continue
            crop_img = img[rand_h:rand_h + crop_h, rand_w:rand_w + crop_w]
            K_new = K.copy()
            K_new[0, 2] -= rand_w
            K_new[1, 2] -= rand_h
            return crop_img, crop_mask, K_new

    def get_total_frames(self):
        return len(self.framelist)

    def __len__(self):
        return self.get_total_frames()

    def __getitem__(self, idx):
        frame_name = self.framelist[idx]
        results = {
            'frame_name': frame_name
        }

        if self.bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')
        results['bgcolor'] = bgcolor / 255.

        if self.prefetch:
            img, alpha, orig_W, orig_H = self.preload[frame_name]
            img = alpha * img + (1.0 - alpha) * bgcolor[None, None, :]
        else:
            img, alpha, orig_W, orig_H = self.load_image(frame_name, bgcolor)
        img = (img / 255.).astype('float32')

        dst_skel_info = self.query_dst_skeleton(frame_name)
        dst_poses = dst_skel_info['poses']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']

        assert frame_name in self.cameras
        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        if 'target_size' in self.cfg:
            scale_w, scale_h = self.cfg['target_size'][0] / orig_W, self.cfg['target_size'][1] / orig_H
        else:
            scale_w, scale_h = self.cfg['resize_img_scale']
        K[:1] *= scale_w
        K[1:2] *= scale_h

        E = self.cameras[frame_name]['extrinsics']
        E, global_tfms = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'],
                return_global_tfms=True)
        R = E[:3, :3]
        T = E[:3, 3]
        results.update({
            'global_tfms': global_tfms
        })

        if self.cfg['crop_size'] != [-1, -1]:
            img, alpha, K = self.crop_image(img, alpha, K)

        results.update({
            'K': K.astype(np.float32),
            'E': E.astype(np.float32),
        })

        results['target_rgbs'] = img
        results['target_masks'] = alpha[:, :, 0].astype(np.float32)

        dst_Rs, dst_Ts = body_pose_to_body_RTs(
                dst_poses, dst_tpose_joints
            )
        cnl_gtfms = get_canonical_global_tfms(self.canonical_joints)
        results.update({
            'dst_poses': dst_poses,
            'dst_Rs': dst_Rs,
            'dst_Ts': dst_Ts,
            'cnl_gtfms': cnl_gtfms
        })

        # 1. ignore global orientation
        # 2. add a small value to avoid all zeros
        dst_posevec_69 = dst_poses.reshape(-1)[3:] + 1e-2
        results.update({
            'dst_posevec': dst_posevec_69,
        })

        results.update({
            'joints': get_joints_from_pose(dst_poses, dst_tpose_joints),
            'dst_tpose_joints': dst_tpose_joints,
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
        for idx in range(self.get_total_frames()):
            frame_name = self.framelist[idx]

            dst_skel_info = self.query_dst_skeleton(frame_name)

            E = self.cameras[frame_name]['extrinsics']
            E, global_tfms = apply_global_tfm_to_camera(
                E=E,
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'],
                return_global_tfms=True)

            Es.append(E)
        return np.stack(Es, axis=0)
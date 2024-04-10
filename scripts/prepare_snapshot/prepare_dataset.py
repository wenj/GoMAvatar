import os
import sys
sys.path.append('../../')
import cv2
import h5py
import trimesh
import seaborn as sns

from shutil import copyfile

import pickle
import yaml
import numpy as np
from tqdm import tqdm

from pathlib import Path

from utils.smpl.smpl_numpy import SMPL
from utils.file_util import split_path
from utils.image_util import load_image, save_image, to_3ch_image
from utils.body_util import _rvec_to_rmtx

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    'female-3-casual.yaml',
                    'the path of config file')

MODEL_DIR = '../../utils/smpl/models'


def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config


def prepare_dir(output_path, name):
    out_dir = os.path.join(output_path, name)
    os.makedirs(out_dir, exist_ok=True)

    return out_dir


def get_mask(subject_dir, img_name):
    msk_path = os.path.join(subject_dir, 'mask',
                            img_name)[:-4] + '.png'
    msk = np.array(load_image(msk_path))[:, :, 0]
    msk = (msk != 0).astype(np.uint8)

    msk_path = os.path.join(subject_dir, 'mask_cihp',
                            img_name)[:-4] + '.png'
    msk_cihp = np.array(load_image(msk_path))[:, :, 0]
    msk_cihp = (msk_cihp != 0).astype(np.uint8)

    msk = (msk | msk_cihp).astype(np.uint8)
    msk[msk == 1] = 255

    return msk


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()


def get_KRTD(camera):
    K = np.zeros([3, 3])
    K[0, 0] = camera['camera_f'][0]
    K[1, 1] = camera['camera_f'][1]
    K[:2, 2] = camera['camera_c']
    K[2, 2] = 1
    R = np.eye(3)
    T = np.zeros([3])
    D = camera['camera_k']
    return K, R, T, D


def extract_image(data_path):
    cap = cv2.VideoCapture(data_path)

    ret, frame = cap.read()
    i = 0

    imgs = []
    while ret:
        imgs.append(frame)
        ret, frame = cap.read()
        i = i + 1

    cap.release()

    return imgs


def load_h5py(fpath):
    return h5py.File(fpath, "r")


def main(argv):
    del argv  # Unused.

    cfg = parse_config()
    subject = cfg['dataset']['subject']
    sex = cfg['dataset']['sex']
    start_frame = cfg['start_frame']
    end_frame = cfg['end_frame']
    skip = cfg['skip']

    dataset_dir = cfg['dataset']['snapshot_path']
    subject_dir = os.path.join(dataset_dir, subject)
    smpl_params_dir = os.path.join(cfg['dataset']['pose_path'], subject, 'poses', 'anim_nerf_{}.npz'.format(cfg['split']))
    camera_path = os.path.join(subject_dir, 'camera.pkl')
    camera = read_pickle(camera_path)
    K, R, T, D = get_KRTD(camera)
    E = np.eye(4)  # (4, 4)
    E[:3, :3] = R
    E[:3, 3] = T

    output_path = os.path.join(cfg['output']['dir'],
                               subject if 'name' not in cfg['output'].keys() else cfg['output']['name'])
    os.makedirs(output_path, exist_ok=True)
    out_img_dir = prepare_dir(output_path, 'images')
    out_mask_dir = prepare_dir(output_path, 'masks')

    # process video
    video_path = os.path.join(subject_dir, subject + '.mp4')
    imgs = extract_image(video_path)

    # process mask
    mask_path = os.path.join(subject_dir, 'masks.hdf5')
    masks = np.asarray(load_h5py(mask_path)["masks"]).astype(np.uint8)
    # masks = extract_mask(masks)

    # copy config file
    copyfile(FLAGS.cfg, os.path.join(output_path, 'config.yaml'))

    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)

    cameras = {}
    mesh_infos = {}
    all_betas = []

    smpl = dict(np.load(smpl_params_dir))
    betas = smpl['betas'][0]
    global_orient_all = smpl['global_orient']
    pose_all = smpl['body_pose']
    trans_all = smpl['transl']

    for idx in tqdm(range(start_frame, end_frame + 1, skip)):
        out_name = 'frame_{:06d}'.format((idx - start_frame) // skip)

        img = imgs[idx]
        img = cv2.undistort(img, K, D)
        img = cv2.resize(img, dsize=None, fx=1 / 2, fy=1 / 2)
        mask = masks[idx]
        mask = cv2.undistort(mask, K, D)
        mask = cv2.resize(mask, dsize=None, fx=1 / 2, fy=1 / 2)

        cv2.imwrite(os.path.join(out_img_dir, out_name + '.png'), img)
        mask = (mask * 255.).astype(np.uint8)
        cv2.imwrite(os.path.join(out_mask_dir, out_name + '.png'), mask)

        poses = pose_all[(idx - start_frame) // skip]  # (72,)
        poses = np.concatenate([np.zeros([3], dtype=poses.dtype), poses])

        Rh = global_orient_all[(idx - start_frame) // skip]  # (3,)
        Th = trans_all[(idx - start_frame) // skip]  # (3,)

        _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
        pelvis_pos = tpose_joints[0].copy()

        _, joints = smpl_model(poses, betas)
        Th = Th + tpose_joints[0] - _rvec_to_rmtx(Rh) @ tpose_joints[0]

        # tpose_joints -= pelvis_pos[None, :]
        # joints -= pelvis_pos[None, :]

        mesh_infos[out_name] = {
            'Rh': Rh,
            'Th': Th,
            'poses': poses,
            'joints': joints,
            'tpose_joints': tpose_joints
        }

        # write camera info
        K_new = np.array(K)
        K_new[:2] *= .5
        cameras[out_name] = {
            'intrinsics': K_new,
            'extrinsics': E,
            'distortions': D
        }

    # write camera infos
    with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:
        pickle.dump(cameras, f)

    # write mesh infos
    with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:
        pickle.dump(mesh_infos, f)

    # write canonical joints
    avg_betas = betas
    smpl_model = SMPL(sex, model_dir=MODEL_DIR)
    v, template_joints, weights = smpl_model(np.zeros(72), avg_betas, return_weights=True)
    faces = smpl_model.faces

    # pelvis_pos = template_joints[0].copy()
    # template_joints = template_joints - pelvis_pos[None, :]
    # v = v - pelvis_pos[None, :]

    mesh = trimesh.Trimesh(v, faces=faces)
    edges = mesh.edges
    with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:
        pickle.dump(
            {
                'vertex': v,
                'joints': template_joints,
                'weights': weights,
                'edges': edges,
                'faces': faces
            }, f)


if __name__ == '__main__':
    app.run(main)

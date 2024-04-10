from math import cos, sin

import numpy as np

import torch

EPS = 1e-3

SMPL_JOINT_IDX = {
    'pelvis_root': 0,
    'left_hip': 1,
    'right_hip': 2,
    'belly_button': 3,
    'left_knee': 4,
    'right_knee': 5,
    'lower_chest': 6,
    'left_ankle': 7,
    'right_ankle': 8,
    'upper_chest': 9,
    'left_toe': 10,
    'right_toe': 11,
    'neck': 12,
    'left_clavicle': 13,
    'right_clavicle': 14,
    'head': 15,
    'left_shoulder': 16,
    'right_shoulder': 17,
    'left_elbow': 18,
    'right_elbow': 19,
    'left_wrist': 20,
    'right_wrist': 21,
    'left_thumb': 22,
    'right_thumb': 23
}

SMPL_PARENT = {
    1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 
    11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 
    21: 19, 22: 20, 23: 21}

SMPLX_JOINT_NAMES = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "jaw": 22,
    "left_eye_smplhf": 23,
    "right_eye_smplhf": 24,
    "left_index1": 25,
    "left_index2": 26,
    "left_index3": 27,
    "left_middle1": 28,
    "left_middle2": 29,
    "left_middle3": 30,
    "left_pinky1": 31,
    "left_pinky2": 32,
    "left_pinky3": 33,
    "left_ring1": 34,
    "left_ring2": 35,
    "left_ring3": 36,
    "left_thumb1": 37,
    "left_thumb2": 38,
    "left_thumb3": 39,
    "right_index1": 40,
    "right_index2": 41,
    "right_index3": 42,
    "right_middle1": 43,
    "right_middle2": 44,
    "right_middle3": 45,
    "right_pinky1": 46,
    "right_pinky2": 47,
    "right_pinky3": 48,
    "right_ring1": 49,
    "right_ring2": 50,
    "right_ring3": 51,
    "right_thumb1": 52,
    "right_thumb2": 53,
    "right_thumb3": 54,
}

SMPLX_PARENT = {
    0: -1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8,
    12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19,
    22: 15, 23: 15, 24: 15, 25: 20, 26: 25, 27: 26, 28: 20, 29: 28, 30: 29, 31: 20,
    32: 31, 33: 32, 34: 20, 35: 34, 36: 35, 37: 20, 38: 37, 39: 38, 40: 21, 41: 40,
    42: 41, 43: 21, 44: 43, 45: 44, 46: 21, 47: 46, 48: 47, 49: 21, 50: 49, 51: 50,
    52: 21, 53: 52, 54: 53}

TORSO_JOINTS_NAME = [
    'pelvis_root', 'belly_button', 'lower_chest', 'upper_chest', 'left_clavicle', 'right_clavicle'
]
TORSO_JOINTS = [
    SMPL_JOINT_IDX[joint_name] for joint_name in TORSO_JOINTS_NAME
]
BONE_STDS = np.array([0.03, 0.06, 0.03])
HEAD_STDS = np.array([0.06, 0.06, 0.06])
JOINT_STDS = np.array([0.02, 0.02, 0.02])


def _to_skew_matrix(v):
    r""" Compute the skew matrix given a 3D vectors.

    Args:
        - v: Array (3, )

    Returns:
        - Array (3, 3)

    """
    vx, vy, vz = v.ravel()
    return np.array([[0, -vz, vy],
                    [vz, 0, -vx],
                    [-vy, vx, 0]])


def _to_skew_matrix_tensor(v):
    r""" Compute the skew matrix given a 3D vectors.

    Args:
        - v: Array (3, )

    Returns:
        - Array (3, 3)

    """
    mat = v.new_zeros([3, 3])
    vx, vy, vz = v[0], v[1], v[2]
    mat[0, 0], mat[0, 1], mat[0, 2] = 0, -vz, vy
    mat[1, 0], mat[1, 1], mat[1, 2] = vz, 0, -vx
    mat[2, 0], mat[2, 1], mat[2, 2] = -vy, vx, 0
    return mat


def _to_skew_matrices(batch_v):
    r""" Compute the skew matrix given 3D vectors. (batch version)

    Args:
        - batch_v: Array (N, 3)

    Returns:
        - Array (N, 3, 3)

    """
    batch_size = batch_v.shape[0]
    skew_matrices = np.zeros(shape=(batch_size, 3, 3), dtype=np.float32)

    for i in range(batch_size):
        skew_matrices[i] = _to_skew_matrix(batch_v[i])

    return skew_matrices


def _get_rotation_mtx(v1, v2):
    r""" Compute the rotation matrices between two 3D vector. (batch version)
    
    Args:
        - v1: Array (N, 3)
        - v2: Array (N, 3)

    Returns:
        - Array (N, 3, 3)

    Reference:
        https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """

    batch_size = v1.shape[0]
    
    v1 = v1 / np.clip(np.linalg.norm(v1, axis=-1, keepdims=True), 1e-5, None)
    v2 = v2 / np.clip(np.linalg.norm(v2, axis=-1, keepdims=True), 1e-5, None)
    
    normal_vec = np.cross(v1, v2, axis=-1)
    cos_v = np.zeros(shape=(batch_size, 1))
    for i in range(batch_size):
        cos_v[i] = v1[i].dot(v2[i])

    skew_mtxs = _to_skew_matrices(normal_vec)
    
    Rs = np.zeros(shape=(batch_size, 3, 3), dtype=np.float32)
    for i in range(batch_size):
        Rs[i] = np.eye(3) + skew_mtxs[i] + \
                    (skew_mtxs[i].dot(skew_mtxs[i])) * (1./(1. + cos_v[i]))
    
    return Rs


def _construct_G(R_mtx, T):
    r""" Build 4x4 [R|T] matrix from rotation matrix, and translation vector
    
    Args:
        - R_mtx: Array (3, 3)
        - T: Array (3,)

    Returns:
        - Array (4, 4)
    """

    G = np.array(
        [[R_mtx[0, 0], R_mtx[0, 1], R_mtx[0, 2], T[0]],
         [R_mtx[1, 0], R_mtx[1, 1], R_mtx[1, 2], T[1]],
         [R_mtx[2, 0], R_mtx[2, 1], R_mtx[2, 2], T[2]],
         [0.,          0.,          0.,          1.]],
        dtype='float32')

    return G
    

def _deform_gaussian_volume(
        grid_size, 
        bbox_min_xyz,
        bbox_max_xyz,
        center, 
        scale_mtx, 
        rotation_mtx):
    r""" Deform a standard Gaussian volume.
    
    Args:
        - grid_size:    Integer
        - bbox_min_xyz: Array (3, )
        - bbox_max_xyz: Array (3, )
        - center:       Array (3, )   - center of Gaussain to be deformed
        - scale_mtx:    Array (3, 3)  - scale of Gaussain to be deformed
        - rotation_mtx: Array (3, 3)  - rotation matrix of Gaussain to be deformed

    Returns:
        - Array (grid_size, grid_size, grid_size)
    """

    R = rotation_mtx
    S = scale_mtx

    # covariance matrix after scaling and rotation
    SIGMA = R.dot(S).dot(S).dot(R.T)

    min_x, min_y, min_z = bbox_min_xyz
    max_x, max_y, max_z = bbox_max_xyz
    zgrid, ygrid, xgrid = np.meshgrid(
        np.linspace(min_z, max_z, grid_size),
        np.linspace(min_y, max_y, grid_size),
        np.linspace(min_x, max_x, grid_size),
        indexing='ij')
    grid = np.stack([xgrid - center[0], 
                     ygrid - center[1], 
                     zgrid - center[2]],
                    axis=-1)

    dist = np.einsum('abci, abci->abc', np.einsum('abci, ij->abcj', grid, SIGMA), grid)

    return np.exp(-1 * dist)


def _std_to_scale_mtx(stds):
    r""" Build scale matrix from standard deviations
    
    Args:
        - stds: Array(3,)

    Returns:
        - Array (3, 3)
    """

    scale_mtx = np.eye(3, dtype=np.float32)
    scale_mtx[0][0] = 1.0/stds[0]
    scale_mtx[1][1] = 1.0/stds[1]
    scale_mtx[2][2] = 1.0/stds[2]

    return scale_mtx


def _rvec_to_rmtx(rvec):
    r''' apply Rodriguez Formula on rotate vector (3,)

    Args:
        - rvec: Array (3,)

    Returns:
        - Array (3, 3)
    '''
    rvec = rvec.reshape(3, 1)

    norm = np.linalg.norm(rvec)
    theta = norm
    r = rvec / (norm + 1e-5)

    skew_mtx = _to_skew_matrix(r)

    return cos(theta)*np.eye(3) + \
           sin(theta)*skew_mtx + \
           (1-cos(theta))*r.dot(r.T)


def _rvec_to_rmtx_tensor(rvec):
    r''' apply Rodriguez Formula on rotate vector (3,)

    Args:
        - rvec: Array (3,)

    Returns:
        - Array (3, 3)
    '''
    rvec = rvec.reshape(3, 1)

    norm = torch.linalg.norm(rvec)
    theta = norm
    r = rvec / (norm + 1e-5)

    skew_mtx = _to_skew_matrix_tensor(r)

    return cos(theta)*torch.eye(3, device=theta.device) + \
           sin(theta)*skew_mtx + \
           (1-cos(theta))*(r @ r.T)


def body_pose_to_body_RTs(jangles, tpose_joints, use_smplx=False):
    r""" Convert body pose to global rotation matrix R and translation T.

    Args:
        - jangles (joint angles): Array (Total_Joints x 3, )
        - tpose_joints:           Array (Total_Joints, 3)

    Returns:
        - Rs: Array (Total_Joints, 3, 3)
        - Ts: Array (Total_Joints, 3)
    """

    if not use_smplx:
        PARENT = SMPL_PARENT
    else:
        PARENT = SMPLX_PARENT

    jangles = jangles.reshape(-1, 3)
    total_joints = jangles.shape[0]
    assert tpose_joints.shape[0] == total_joints

    Rs = np.zeros(shape=[total_joints, 3, 3], dtype='float32')
    Rs[0] = _rvec_to_rmtx(jangles[0, :])

    Ts = np.zeros(shape=[total_joints, 3], dtype='float32')
    Ts[0] = tpose_joints[0, :]

    for i in range(1, total_joints):
        Rs[i] = _rvec_to_rmtx(jangles[i, :])
        Ts[i] = tpose_joints[i, :] - tpose_joints[PARENT[i], :]

    return Rs, Ts


def body_pose_to_body_RTs_tensor(jangles, tpose_joints, use_smplx=False):
    r""" Convert body pose to global rotation matrix R and translation T.

    Args:
        - jangles (joint angles): Array (Total_Joints x 3, )
        - tpose_joints:           Array (Total_Joints, 3)

    Returns:
        - Rs: Array (Total_Joints, 3, 3)
        - Ts: Array (Total_Joints, 3)
    """

    if not use_smplx:
        PARENT = SMPL_PARENT
    else:
        PARENT = SMPLX_PARENT

    jangles = jangles.reshape(-1, 3)
    total_joints = jangles.shape[0]
    assert tpose_joints.shape[0] == total_joints

    Rs = tpose_joints.new_zeros([total_joints, 3, 3])
    Rs[0] = _rvec_to_rmtx_tensor(jangles[0, :])

    Ts = tpose_joints.new_zeros([total_joints, 3])
    Ts[0] = tpose_joints[0, :]

    for i in range(1, total_joints):
        Rs[i] = _rvec_to_rmtx_tensor(jangles[i, :])
        Ts[i] = tpose_joints[i, :] - tpose_joints[PARENT[i], :]

    return Rs, Ts


def get_canonical_global_tfms(canonical_joints, use_smplx=False):
    r""" Convert canonical joints to 4x4 global transformation matrix.
    
    Args:
        - canonical_joints: Array (Total_Joints, 3)

    Returns:
        - Array (Total_Joints, 4, 4)
    """
    if not use_smplx:
        PARENT = SMPL_PARENT
    else:
        PARENT = SMPLX_PARENT

    total_bones = canonical_joints.shape[0]

    gtfms = np.zeros(shape=(total_bones, 4, 4), dtype='float32')
    gtfms[0] = _construct_G(np.eye(3), canonical_joints[0,:])

    for i in range(1, total_bones):
        translate = canonical_joints[i,:] - canonical_joints[PARENT[i],:]
        gtfms[i] = gtfms[PARENT[i]].dot(
                            _construct_G(np.eye(3), translate))

    return gtfms


def approx_gaussian_bone_volumes(
    tpose_joints, 
    bbox_min_xyz, bbox_max_xyz,
    grid_size=32,
    use_smplx=False
):
    r""" Compute approximated Gaussian bone volume.
    
    Args:
        - tpose_joints:  Array (Total_Joints, 3)
        - bbox_min_xyz:  Array (3, )
        - bbox_max_xyz:  Array (3, )
        - grid_size:     Integer
        - has_bg_volume: boolean

    Returns:
        - Array (Total_Joints + 1, 3, 3, 3)
    """
    if not use_smplx:
        PARENT = SMPL_PARENT
    else:
        PARENT = SMPLX_PARENT

    total_joints = tpose_joints.shape[0]

    grid_shape = [grid_size] * 3
    tpose_joints = tpose_joints.astype(np.float32)

    calibrated_bone = np.array([0.0, 1.0, 0.0], dtype=np.float32)[None, :]
    g_volumes = []
    for joint_idx in range(0, total_joints):
        gaussian_volume = np.zeros(shape=grid_shape, dtype='float32')

        is_parent_joint = False
        for bone_idx, parent_idx in PARENT.items():
            if joint_idx != parent_idx:
                continue

            S = _std_to_scale_mtx(BONE_STDS * 2.)
            if joint_idx in TORSO_JOINTS:
                S[0][0] *= 1/1.5
                S[2][2] *= 1/1.5

            start_joint = tpose_joints[PARENT[bone_idx]]
            end_joint = tpose_joints[bone_idx]
            target_bone = (end_joint - start_joint)[None, :]

            R = _get_rotation_mtx(calibrated_bone, target_bone)[0].astype(np.float32)

            center = (start_joint + end_joint) / 2.0

            bone_volume = _deform_gaussian_volume(
                            grid_size, 
                            bbox_min_xyz,
                            bbox_max_xyz,
                            center, S, R)
            gaussian_volume = gaussian_volume + bone_volume

            is_parent_joint = True

        if not is_parent_joint:
            # The joint is not other joints' parent, meaning it is an end joint
            joint_stds = HEAD_STDS if joint_idx == SMPL_JOINT_IDX['head'] else JOINT_STDS
            S = _std_to_scale_mtx(joint_stds * 2.)

            center = tpose_joints[joint_idx]
            gaussian_volume = _deform_gaussian_volume(
                                grid_size, 
                                bbox_min_xyz,
                                bbox_max_xyz,
                                center, 
                                S, 
                                np.eye(3, dtype='float32'))
            
        g_volumes.append(gaussian_volume)
    g_volumes = np.stack(g_volumes, axis=0)

    # concatenate background weights
    bg_volume = 1.0 - np.sum(g_volumes, axis=0, keepdims=True).clip(min=0.0, max=1.0)
    g_volumes = np.concatenate([g_volumes, bg_volume], axis=0)
    g_volumes = g_volumes / np.sum(g_volumes, axis=0, keepdims=True).clip(min=0.001)
    
    return g_volumes


def approx_gaussian_bone_volumes_smpl(
        vertex,
        weights_init,
        xyzs,
        K=1,
        sigma=0.2,
):
    r""" Compute approximated Gaussian bone volume based on smpl init

    Args:
        - vertex:  Array (N_smpl, 3)
        - weights_init: Array (N_smpl, Total_Joints)
        - xyzs: Array (3, N)

    Returns:
        - Array (Total_Joints + 1, 3, 3, 3)
    """
    _, N = xyzs.shape
    _, total_joints = weights_init.shape

    g_volumes = xyzs.new_zeros([total_joints, N])
    block_size = 10000
    for i in range(0, N, block_size):
        # print(i)
        end_idx = min(N, i + block_size)
        xyzs_single = xyzs[:, i:end_idx]
        dist = torch.sum((xyzs_single[:, :, None] - vertex.permute(1, 0)[:, None, :]) ** 2, dim=0) # N x N_smpl
        topk = torch.topk(dist, dim=-1, k=K, largest=False) # N x K
        idx = topk.indices
        dist = topk.values
        prob = torch.exp(-0.5 * dist / (sigma * sigma)) # N x K
        weights_init_single = weights_init[idx.reshape(-1)].reshape(end_idx - i, K, -1)
        g_volumes[:, i:end_idx] = torch.sum(prob[None] * weights_init_single.permute(2, 0, 1), dim=-1) / K # N_joint x N
    # concatenate background weights
    bg_volume = 1.0 - torch.sum(g_volumes, dim=0, keepdim=True).clip(min=0.0, max=1.0)
    g_volumes = torch.cat([g_volumes, bg_volume], dim=0)
    g_volumes = g_volumes / torch.sum(g_volumes, dim=0, keepdim=True).clip(min=0.001)

    return g_volumes


def get_joints_from_pose(poses, tpose_joints, use_smplx=False):
    if not use_smplx:
        PARENT = SMPL_PARENT
    else:
        PARENT = SMPLX_PARENT

    N = tpose_joints.shape[0]
    poses = poses.reshape(N, -1)

    transforms = np.eye(4)[None, :, :].repeat(N, axis=0).astype(np.float32)

    transforms[0, :3, :3] = _rvec_to_rmtx(poses[0])
    transforms[0, :3, 3] = tpose_joints[0]

    for i, (tpose_joint, pose) in enumerate(zip(tpose_joints, poses)):
        if i == 0:
            continue
        transforms[i, :3, :3] = _rvec_to_rmtx(pose)
        # print(i, SMPL_PARENT[i], tpose_joint, tpose_joints[SMPL_PARENT[i]], tpose_joint - tpose_joints[SMPL_PARENT[i]])
        transforms[i, :3, 3] = tpose_joint - tpose_joints[PARENT[i]]

    # print(poses)
    # print(transforms)

    joints = np.zeros([N, 4]).astype(np.float32)
    joints[0] = transforms[0] @ np.array([0, 0, 0, 1]).astype(np.float32)
    for i in range(1, N):
        transforms[i] = transforms[PARENT[i]] @ transforms[i]
        joints[i] = transforms[i] @ np.array([0, 0, 0, 1]).astype(np.float32)
    return joints[:, :3] / joints[:, 3:]


def get_joints_from_RTs(tpose_joints, cnl_gtfms, dst_Rs, dst_Ts, use_smplx=False):
    global_Rs, global_Ts = get_global_RTs(cnl_gtfms, dst_Rs, dst_Ts, use_smplx=use_smplx)
    joints = (global_Rs @ tpose_joints[..., None]).squeeze(-1) + global_Ts
    return joints


def _construct_G_tensor(R_mtx, T):
    r''' Tile ration matrix and translation vector to build a 4x4 matrix.

	Args:
		R_mtx: Tensor (B, TOTAL_BONES, 3, 3)
		T:     Tensor (B, TOTAL_BONES, 3)

	Returns:
		G:     Tensor (B, TOTAL_BONES, 4, 4)
	'''
    batch_size, total_bones = R_mtx.shape[:2]

    G = torch.zeros(size=(batch_size, total_bones, 4, 4),
                    dtype=R_mtx.dtype, device=R_mtx.device)
    G[:, :, :3, :3] = R_mtx
    G[:, :, :3, 3] = T
    G[:, :, 3, 3] = 1.0

    return G


def get_global_RTs(cnl_gtfms, dst_Rs, dst_Ts, use_smplx=False):
    if not use_smplx:
        PARENT = SMPL_PARENT
    else:
        PARENT = SMPLX_PARENT

    total_bones = cnl_gtfms.shape[1]
    dst_gtfms = torch.zeros_like(cnl_gtfms)

    local_Gs = _construct_G_tensor(dst_Rs, dst_Ts)
    dst_gtfms[:, 0, :, :] = local_Gs[:, 0, :, :]

    for i in range(1, total_bones):
        dst_gtfms[:, i, :, :] = torch.matmul(
            dst_gtfms[:, PARENT[i],
            :, :].clone(),
            local_Gs[:, i, :, :])

    dst_gtfms = dst_gtfms.view(cnl_gtfms.shape[0], -1, 4, 4)

    f_mtx = torch.matmul(dst_gtfms, torch.inverse(cnl_gtfms))
    f_mtx = f_mtx.view(-1, total_bones, 4, 4)

    scale_Rs = f_mtx[:, :, :3, :3]
    Ts = f_mtx[:, :, :3, 3]

    return scale_Rs, Ts


def apply_lbs(xyzs_canonical, global_Rs, global_Ts, lbs_weights):
    xyzs_trans = torch.einsum('b...ij,bjk->b...ik', global_Rs, xyzs_canonical) + global_Ts[:, :, :, None]
    xyzs_sum = torch.sum(xyzs_trans * lbs_weights[:-1][None, :, None, :], dim=1)
    return xyzs_sum


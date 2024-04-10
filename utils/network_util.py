import math

import numpy as np
import torch
import torch.nn as nn

from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.ops.knn import knn_points

from .pc_util import img_T_world

# import sys
# sys.path.append('/home/jw116/humannerf_clean/Pointcept')
# from lib.pointops.functions import pointops
# import pointops

###############################################################################
## Network Components - Convolutional Decoders
###############################################################################

class ConvDecoder3D(nn.Module):
    r""" Convolutional 3D volume decoder."""

    def __init__(self, embedding_size=256, volume_size=128, voxel_channels=4):
        r""" 
            Args:
                embedding_size: integer
                volume_size: integer
                voxel_channels: integer
        """    
        super(ConvDecoder3D, self).__init__()

        self.block_mlp = nn.Sequential(nn.Linear(embedding_size, 1024), 
                                       nn.LeakyReLU(0.2))
        block_conv = []
        inchannels, outchannels = 1024, 512
        for _ in range(int(np.log2(volume_size)) - 1):
            block_conv.append(nn.ConvTranspose3d(inchannels, 
                                                 outchannels, 
                                                 4, 2, 1))
            block_conv.append(nn.LeakyReLU(0.2))
            if inchannels == outchannels:
                outchannels = inchannels // 2
            else:
                inchannels = outchannels
        block_conv.append(nn.ConvTranspose3d(inchannels, 
                                             voxel_channels, 
                                             4, 2, 1))
        self.block_conv = nn.Sequential(*block_conv)

        for m in [self.block_mlp, self.block_conv]:
            initseq(m)

    def forward(self, embedding):
        """ 
            Args:
                embedding: Tensor (B, N)
        """    
        return self.block_conv(self.block_mlp(embedding).view(-1, 1024, 1, 1, 1))


###############################################################################
## Network Components - 3D rotations
###############################################################################

class RodriguesModule(nn.Module):
    def forward(self, rvec):
        r''' Apply Rodriguez formula on a batch of rotation vectors.

            Args:
                rvec: Tensor (B, 3)
            
            Returns
                rmtx: Tensor (B, 3, 3)
        '''
        theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        costh = torch.cos(theta)
        sinth = torch.sin(theta)
        return torch.stack((
            rvec[:, 0] ** 2 + (1. - rvec[:, 0] ** 2) * costh,
            rvec[:, 0] * rvec[:, 1] * (1. - costh) - rvec[:, 2] * sinth,
            rvec[:, 0] * rvec[:, 2] * (1. - costh) + rvec[:, 1] * sinth,

            rvec[:, 0] * rvec[:, 1] * (1. - costh) + rvec[:, 2] * sinth,
            rvec[:, 1] ** 2 + (1. - rvec[:, 1] ** 2) * costh,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) - rvec[:, 0] * sinth,

            rvec[:, 0] * rvec[:, 2] * (1. - costh) - rvec[:, 1] * sinth,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) + rvec[:, 0] * sinth,
            rvec[:, 2] ** 2 + (1. - rvec[:, 2] ** 2) * costh), 
        dim=1).view(-1, 3, 3)


###############################################################################
## Network Components - compute motion base
###############################################################################


SMPL_PARENT = {
    1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 
    11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 
    21: 19, 22: 20, 23: 21}


class MotionBasisComputer(nn.Module):
    r"""Compute motion bases between the target pose and canonical pose."""

    def __init__(self, total_bones=24):
        super(MotionBasisComputer, self).__init__()
        self.total_bones = total_bones

    def _construct_G(self, R_mtx, T):
        r''' Tile ration matrix and translation vector to build a 4x4 matrix.

        Args:
            R_mtx: Tensor (B, TOTAL_BONES, 3, 3)
            T:     Tensor (B, TOTAL_BONES, 3)

        Returns:
            G:     Tensor (B, TOTAL_BONES, 4, 4)
        '''
        batch_size, total_bones = R_mtx.shape[:2]
        assert total_bones == self.total_bones

        G = torch.zeros(size=(batch_size, total_bones, 4, 4),
                        dtype=R_mtx.dtype, device=R_mtx.device)
        G[:, :, :3, :3] = R_mtx
        G[:, :, :3, 3] = T
        G[:, :, 3, 3] = 1.0
    
        return G

    def forward(self, dst_Rs, dst_Ts, cnl_gtfms):
        r"""
        Args:
            dst_Rs:    Tensor (B, TOTAL_BONES, 3, 3)
            dst_Ts:    Tensor (B, TOTAL_BONES, 3)
            cnl_gtfms: Tensor (B, TOTAL_BONES, 4, 4)
                
        Returns:
            scale_Rs: Tensor (B, TOTAL_BONES, 3, 3)
            Ts:       Tensor (B, TOTAL_BONES, 3)
        """
        dst_gtfms = torch.zeros_like(cnl_gtfms)

        local_Gs = self._construct_G(dst_Rs, dst_Ts)    
        dst_gtfms[:, 0, :, :] = local_Gs[:, 0, :, :]

        for i in range(1, self.total_bones):
            dst_gtfms[:, i, :, :] = torch.matmul(
                                        dst_gtfms[:, SMPL_PARENT[i], 
                                                  :, :].clone(),
                                        local_Gs[:, i, :, :])

        dst_gtfms = dst_gtfms.view(-1, 4, 4)
        inv_dst_gtfms = torch.inverse(dst_gtfms)
        
        cnl_gtfms = cnl_gtfms.view(-1, 4, 4)
        f_mtx = torch.matmul(cnl_gtfms, inv_dst_gtfms)
        f_mtx = f_mtx.view(-1, self.total_bones, 4, 4)

        scale_Rs = f_mtx[:, :, :3, :3]
        Ts = f_mtx[:, :, :3, 3]

        return scale_Rs, Ts


class TVLoss(nn.Module):
    def __init__(self, res):
        super(TVLoss, self).__init__()
        self.res = res

    def forward(self, x):
        x = x.reshape(x.shape[0], self.res, self.res, self.res)
        c, h, w, d = x.shape

        count_h = x[:, 1:, :, :].numel()
        count_w = x[:, :, 1:, :].numel()
        count_d = x[:, :, :, 1:].numel()
        h_tv = torch.pow((x[:, 1:, :, :] - x[:, :h-1, :, :]), 2).sum()
        w_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :w-1, :]), 2).sum()
        d_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :d-1]), 2).sum()
        return (h_tv / count_h + w_tv / count_w + d_tv / count_d) / c


class LaplacianLoss(nn.Module):
    # adapted from https://github.com/snuvclab/chupa/blob/main/src/normal_nds/nds/losses/laplacian.py#L9
    def __init__(self, edges, V):
        super(LaplacianLoss, self).__init__()

        edges = torch.tensor(edges)
        e0, e1 = edges.unbind(1)

        idx01 = torch.stack([e0, e1], dim=1)  # (sum(E_n), 2)
        idx10 = torch.stack([e1, e0], dim=1)  # (sum(E_n), 2)
        idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*sum(E_n))

        # First, we construct the adjacency matrix,
        # i.e. A[i, j] = 1 if (i,j) is an edge, or
        # A[e0, e1] = 1 &  A[e1, e0] = 1
        ones = torch.ones(idx.shape[1], dtype=torch.float32)
        A = torch.sparse.FloatTensor(idx, ones, (V, V))

        # the sum of i-th row of A gives the degree of the i-th vertex
        deg = torch.sparse.sum(A, dim=1).to_dense()

        # We construct the Laplacian matrix by adding the non diagonal values
        # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
        deg0 = deg[e0]
        deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
        deg1 = deg[e1]
        deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
        val = torch.cat([deg0, deg1])
        L = torch.sparse.FloatTensor(idx, val, (V, V))

        # Then we add the diagonal values L[i, i] = -1.
        idx = torch.arange(V)
        idx = torch.stack([idx, idx], dim=0)
        ones = torch.ones(idx.shape[1], dtype=torch.float32)
        L -= torch.sparse.FloatTensor(idx, ones, (V, V))

        self.register_buffer('L', L)

    def forward(self, xyzs):
        if len(xyzs.shape) == 2:
            loss = self.L.mm(xyzs.t())
        else:
            loss = self.L[None].bmm(xyzs.permute(0, 2, 1))
        loss = loss.norm(dim=-1) ** 2
        return loss.mean()


def make_meshgrid(H, W):
    y = torch.arange(0, H)
    x = torch.arange(0, W)
    grid_y, grid_x = torch.meshgrid(y, x)
    meshgrid = torch.stack([grid_x, grid_y], dim=0)
    return meshgrid


def sample_tensor(tensor, n_items, dim=0):
    if tensor.shape[dim] > n_items:
        perm = torch.randperm(tensor.shape[dim]).to(tensor.device)
        idx = perm[:n_items]
        tensor = torch.index_select(tensor, dim, idx)
    return tensor


def mask_to_2dpcd(mask, max_points=np.inf):
    # copied from https://github.com/SamsungLabs/point_based_clothing/blob/b3af6f4b6969170d3dc610e6bc87d7344c6a3402/src/outfit_code/utils.py
    B, H, W = mask.shape
    meshgrid = make_meshgrid(H, W).to(mask.device).float()

    segm_pcds = []
    for i in range(B):
        spcd = meshgrid[:, mask[i] > 0].transpose(1, 0)
        segm_pcds.append(spcd)

    n_spcd = (mask > 0).sum(dim=(1, 2)).min().item()
    n_spcd = min(n_spcd, max_points)

    segm_pcds_sampled = []
    for i in range(B):
        segm_pcd = segm_pcds[i]
        segm_pcd = sample_tensor(segm_pcd, n_spcd)
        segm_pcds_sampled.append(segm_pcd)

    segm_pcds_sampled = torch.stack(segm_pcds_sampled, dim=0)

    return segm_pcds_sampled


class Chamfer2DLoss(nn.Module):
    def __init__(self):
        super(Chamfer2DLoss, self).__init__()

    def forward(self, mask, xyzs, K, E, tb=None):
        _, H, W = mask.shape
        xys = img_T_world(xyzs, K, E).permute(0, 2, 1)
        gt_xys = mask_to_2dpcd(mask)

        if tb is not None:
            tb.summ_pointcloud2d('chamfer/pred', xys, (W, H))
            tb.summ_pointcloud2d('chamfer/gt', gt_xys, (W, H))

        chamfer = chamfer_distance(xys, gt_xys)[0]
        return chamfer


class ChamferLoss(nn.Module):
    def __init__(self, dist_func):
        super(ChamferLoss, self).__init__()

        self.dist_func = dist_func

    def forward(self, x, y):
        x_batch = x.unsqueeze(0)
        y_batch = y.unsqueeze(0)
        x_nn = knn_points(x_batch, y_batch, norm=2, K=1)
        idx_x = x_nn.idx[0, ..., 0]
        y_select = y[idx_x]
        dist_x = self.dist_func(x, y_select).mean()

        y_nn = knn_points(y_batch, x_batch, norm=2, K=1)
        idx_y = y_nn.idx[0, ..., 0]
        x_select = x[idx_y]
        dist_y = self.dist_func(y, x_select).mean()

        return dist_x + dist_y


def chamfer_distance(x, y, dist_func):
    x_batch = x.unsqueeze(0)
    y_batch = y.unsqueeze(0)
    x_nn = knn_points(x_batch, y_batch, norm=2, K=1)
    idx_x = x_nn.idx[0, ..., 0]
    y_select = y[idx_x]
    dist_x = dist_func(x, y_select).mean()

    y_nn = knn_points(y_batch, x_batch, norm=2, K=1)
    idx_y = y_nn.idx[0, ..., 0]
    x_select = x[idx_y]
    dist_y = dist_func(y, x_select).mean()

    return (dist_x + dist_y) * 0.5



###############################################################################
## Init Functions
###############################################################################

def xaviermultiplier(m, gain):
    """ 
        Args:
            m (torch.nn.Module)
            gain (float)

        Returns:
            std (float): adjusted standard deviation
    """ 
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] \
                // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] \
                // m.stride[0] // m.stride[1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std


def xavier_uniform_(m, gain):
    """ Set module weight values with a uniform distribution.

        Args:
            m (torch.nn.Module)
            gain (float)
    """ 
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-(std * math.sqrt(3.0)), std * math.sqrt(3.0))


def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    """ Initialized module weights.

        Args:
            m (torch.nn.Module)
            gain (float)
            weightinitfunc (function)
    """ 
    validclasses = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
                    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    if any([isinstance(m, x) for x in validclasses]):
        weightinitfunc(m, gain)
        if hasattr(m, 'bias'):
            m.bias.data.zero_()

    # blockwise initialization for transposed convs
    if isinstance(m, nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, nn.ConvTranspose3d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]


def initseq(s):
    """ Initialized weights of all modules in a module sequence.

        Args:
            s (torch.nn.Sequential)
    """ 
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)

    initmod(s[-1])


###############################################################################
## misc functions
###############################################################################


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True),
                                      nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                      nn.Linear(mid_planes, mid_planes // share_planes),
                                      nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                      nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k, idx_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v, idx_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1,
                                                                                                          2).contiguous() if i == 1 else layer(
            p_r)  # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes,
                                              self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1,
                                                                                                      2).contiguous() if i % 3 == 0 else layer(
            w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape
        s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x, _ = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes), nn.BatchNorm1d(in_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True))

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1;
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class PointTransformerSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13):
        super().__init__()
        self.c = c
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0],
                                   nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1],
                                   nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2],
                                   nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3],
                                   nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4],
                                   nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True),
                                 nn.Linear(planes[0], k))

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, pxo):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        # x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)
        return x


def pointtransformer_seg_repro(**kwargs):
    model = PointTransformerSeg(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model


def mesh_laplacian_smoothing(meshes, method: str = "uniform"):
    r"""
    Computes the laplacian smoothing objective for a batch of meshes.
    This function supports three variants of Laplacian smoothing,
    namely with uniform weights("uniform"), with cotangent weights ("cot"),
    and cotangent curvature ("cotcurv").For more details read [1, 2].

    Args:
        meshes: Meshes object with a batch of meshes.
        method: str specifying the method for the laplacian.
    Returns:
        loss: Average laplacian smoothing loss across the batch.
        Returns 0 if meshes contains no meshes or all empty meshes.

    Consider a mesh M = (V, F), with verts of shape Nx3 and faces of shape Mx3.
    The Laplacian matrix L is a NxN tensor such that LV gives a tensor of vectors:
    for a uniform Laplacian, LuV[i] points to the centroid of its neighboring
    vertices, a cotangent Laplacian LcV[i] is known to be an approximation of
    the surface normal, while the curvature variant LckV[i] scales the normals
    by the discrete mean curvature. For vertex i, assume S[i] is the set of
    neighboring vertices to i, a_ij and b_ij are the "outside" angles in the
    two triangles connecting vertex v_i and its neighboring vertex v_j
    for j in S[i], as seen in the diagram below.

    .. code-block:: python

               a_ij
                /\
               /  \
              /    \
             /      \
        v_i /________\ v_j
            \        /
             \      /
              \    /
               \  /
                \/
               b_ij

        The definition of the Laplacian is LV[i] = sum_j w_ij (v_j - v_i)
        For the uniform variant,    w_ij = 1 / |S[i]|
        For the cotangent variant,
            w_ij = (cot a_ij + cot b_ij) / (sum_k cot a_ik + cot b_ik)
        For the cotangent curvature, w_ij = (cot a_ij + cot b_ij) / (4 A[i])
        where A[i] is the sum of the areas of all triangles containing vertex v_i.

    There is a nice trigonometry identity to compute cotangents. Consider a triangle
    with side lengths A, B, C and angles a, b, c.

    .. code-block:: python

               c
              /|\
             / | \
            /  |  \
         B /  H|   \ A
          /    |    \
         /     |     \
        /a_____|_____b\
               C

        Then cot a = (B^2 + C^2 - A^2) / 4 * area
        We know that area = CH/2, and by the law of cosines we have

        A^2 = B^2 + C^2 - 2BC cos a => B^2 + C^2 - A^2 = 2BC cos a

        Putting these together, we get:

        B^2 + C^2 - A^2     2BC cos a
        _______________  =  _________ = (B/H) cos a = cos a / sin a = cot a
           4 * area            2CH


    [1] Desbrun et al, "Implicit fairing of irregular meshes using diffusion
    and curvature flow", SIGGRAPH 1999.

    [2] Nealan et al, "Laplacian Mesh Optimization", Graphite 2006.
    """

    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
    weights = 1.0 / weights.float()

    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        if method == "uniform":
            L = meshes.laplacian_packed()
        elif method in ["cot", "cotcurv"]:
            L, inv_areas = cot_laplacian(verts_packed, faces_packed)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                # pyre-fixme[58]: `/` is not supported for operand types `float` and
                #  `Tensor`.
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas
        else:
            raise ValueError("Method should be one of {uniform, cot, cotcurv}")

    if method == "uniform":
        loss = L.mm(verts_packed)
    elif method == "cot":
        # pyre-fixme[61]: `norm_w` is undefined, or not always defined.
        loss = L.mm(verts_packed) * norm_w - verts_packed
    elif method == "cotcurv":
        # pyre-fixme[61]: `norm_w` may not be initialized here.
        loss = (L.mm(verts_packed) - L_sum * verts_packed) * norm_w
    loss = loss.norm(dim=1) ** 2

    # loss = loss * weights
    return loss.mean()


def mesh_color_consistency(color, face_connectivity):
    color0 = color[face_connectivity[:, 0]]
    color1 = color[face_connectivity[:, 1]]
    loss = torch.abs(color0 - color1).mean()
    return loss
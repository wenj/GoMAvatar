import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.transforms.so3 import so3_exp_map

import numpy as np

import torch
import torch.nn as nn

from utils.network_util import initseq


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)

        # get hann window weights
        kick_in_iter = torch.tensor(self.kwargs['kick_in_iter'],
                                    dtype=torch.float32)
        t = torch.clamp(self.kwargs['iter_val'] - kick_in_iter, min=0.)
        N = self.kwargs['full_band_iter'] - kick_in_iter
        m = N_freqs
        alpha = m * t / N

        for freq_idx, freq in enumerate(freq_bands):
            w = (1. - torch.cos(np.pi * torch.clamp(alpha - freq_idx,
                                                    min=0., max=1.))) / 2.
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq, w=w: w * p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, iter_val, kick_in_iter=0, full_band_iter=50000, is_identity=0):
    if is_identity == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': False,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'periodic_fns': [torch.sin, torch.cos],
        'iter_val': iter_val,
        'kick_in_iter': kick_in_iter,
        'full_band_iter': full_band_iter,
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class NonRigidModule(nn.Module):
    """
    this module is copied from humannerf
    """
    def __init__(self, module_cfg, **kwargs):
        super().__init__()

        self.cfg = module_cfg
        self.update_rot = module_cfg.update_rot if hasattr(module_cfg, 'update_rot') else False
        self.update_scale = module_cfg.update_scale if hasattr(module_cfg, 'update_scale') else False

        self.skips = module_cfg.skips
        _, pos_embed_size = get_embedder(module_cfg.multires, module_cfg.i_embed)

        block_mlps = [nn.Linear(pos_embed_size + module_cfg.condition_code_size, module_cfg.mlp_width), nn.ReLU()]

        layers_to_cat_inputs = []
        for i in range(1, module_cfg.mlp_depth):
            if i in self.skips:
                layers_to_cat_inputs.append(len(block_mlps))
                block_mlps += [nn.Linear(module_cfg.mlp_width + pos_embed_size, module_cfg.mlp_width),
                               nn.ReLU()]
            else:
                block_mlps += [nn.Linear(module_cfg.mlp_width, module_cfg.mlp_width), nn.ReLU()]

        output_dim = 3
        if self.update_rot:
            output_dim += 3
        if self.update_scale:
            output_dim += 3
        block_mlps += [nn.Linear(module_cfg.mlp_width, output_dim)]

        self.block_mlps = nn.ModuleList(block_mlps)
        initseq(self.block_mlps)

        self.layers_to_cat_inputs = layers_to_cat_inputs

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope non-rigid offsets are zeros
        if hasattr(module_cfg, 'init_scale'):
            init_val = module_cfg.init_scale
        else:
            init_val = 1e-5
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()

    def forward(self, xyzs_skeleton, dst_posevec, i_iter, R=None, S=None):
        xyzs = xyzs_skeleton.permute(0, 2, 1)
        B, N, _ = xyzs.shape
        pos_embed = get_embedder(self.cfg.multires, i_iter, kick_in_iter=self.cfg.kick_in_iter, full_band_iter=self.cfg.full_band_iter)[0](xyzs)
        h = torch.cat([dst_posevec.unsqueeze(1).repeat(1, N, 1), pos_embed], dim=-1)

        for i in range(len(self.block_mlps)):
            if i in self.layers_to_cat_inputs:
                h = torch.cat([h, pos_embed], dim=-1)
            h = self.block_mlps[i](h)
        offset = h

        xyzs_new = xyzs_skeleton + offset[..., :3].permute(0, 2, 1)
        if self.update_rot:
            B, N, _ = offset.shape
            R_off = so3_exp_map(offset[..., 3:6].reshape(B * N, 3)).reshape(B, N, 3, 3)
            R_new = R_off @ R
        else:
            R_new = R

        if self.update_scale:
            S_new = S + offset[..., 6:]
        else:
            S_new = S

        return xyzs_new, R_new, S_new


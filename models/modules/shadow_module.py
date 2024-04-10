import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.transforms.so3 import so3_exp_map
from pytorch3d import ops

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

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                        freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim


class ShadowModule(nn.Module):
    """
    this module is copied from humannerf
    """

    def __init__(self, module_cfg, **kwargs):
        super().__init__()

        self.cfg = module_cfg

        self.skips = module_cfg.skips
        _, pos_embed_size = get_embedder(module_cfg.multires)

        block_mlps = [nn.Linear(pos_embed_size, module_cfg.mlp_width), nn.ReLU()]

        layers_to_cat_inputs = []
        for i in range(1, module_cfg.mlp_depth):
            if i in self.skips:
                layers_to_cat_inputs.append(len(block_mlps))
                block_mlps += [nn.Linear(module_cfg.mlp_width + pos_embed_size, module_cfg.mlp_width),
                               nn.ReLU()]
            else:
                block_mlps += [nn.Linear(module_cfg.mlp_width, module_cfg.mlp_width), nn.ReLU()]
        self.layers_to_cat_inputs = layers_to_cat_inputs

        output_dim = 1
        block_mlps += [nn.Linear(module_cfg.mlp_width, output_dim)]

        self.block_mlps = nn.ModuleList(block_mlps)
        initseq(self.block_mlps)

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope non-rigid offsets are zeros
        if hasattr(module_cfg, 'init_scale'):
            init_val = module_cfg.init_scale
        else:
            init_val = 1e-5
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()

    def forward(self, normals, **kwargs):
        pos_embed = get_embedder(self.cfg.multires)[0](normals)
        h = pos_embed

        for i in range(len(self.block_mlps)):
            if i in self.layers_to_cat_inputs:
                h = torch.cat([h, pos_embed], dim=-1)
            h = self.block_mlps[i](h)

        return torch.sigmoid(h)

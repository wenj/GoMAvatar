import torch
import torch.nn as nn
import torch.nn.functional as F


class AppearanceModule(nn.Module):
    def __init__(self, module_cfg, canonical_info, **kwargs):
        super().__init__()

        self.cfg = module_cfg

        N = canonical_info['faces'].shape[0]
        color_init = module_cfg.color_init
        self.appearance = nn.Parameter(torch.ones(3, N).float() * color_init)

        bg_col = torch.zeros([3]).float()
        self.register_buffer('bg_col', bg_col)

    def forward(self, **kwargs):
        return self.appearance, self.bg_col

    def set(self, appearance):
        self.appearance = nn.Parameter(appearance)

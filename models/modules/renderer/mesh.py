import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.renderer import (
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRasterizer,
    SoftSilhouetteShader,
    MeshRenderer,
)
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.structures import Meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.blending import hard_rgb_blend

from utils.pc_util import ndc_T_world


def phong_normal_shading(meshes, fragments, vertex_normals) -> torch.Tensor:
    faces = meshes.faces_packed()  # (F, 3)
    faces_normals = vertex_normals[faces]
    ones = torch.ones_like(fragments.bary_coords)
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, ones, faces_normals
    )
    return pixel_normals


def face_normal_shading(meshes, fragments, face_normals) -> torch.Tensor:
    s = fragments.pix_to_face.shape
    pixel_normals = face_normals[fragments.pix_to_face.reshape(-1)].reshape(*s, 3)
    return pixel_normals


class NormalShader(ShaderBase):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardPhongShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments, meshes, normals, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = phong_normal_shading(
            meshes=meshes,
            fragments=fragments,
            vertex_normals=normals,
        )
        images = hard_rgb_blend(colors, fragments, blend_params)
        return images



class Renderer(nn.Module):
    def __init__(self, module_cfg, canonical_info, **kwargs):
        super().__init__()

        self.cfg = module_cfg

        cameras = FoVOrthographicCameras(R=(torch.eye(3, dtype=torch.float32)[None, ...]).cuda(),
                                         T=torch.zeros((1, 3), dtype=torch.float32).cuda(),
                                         znear=[1e-5],
                                         zfar=[1e5],
                                         device='cuda'
                                         )

        if self.cfg.eval_mode:
            raster_settings = RasterizationSettings(
                image_size=(module_cfg.img_size[1], module_cfg.img_size[0]),
                blur_radius=0.0,
                # faces_per_pixel=50,
                max_faces_per_bin=20000,
                bin_size=None,
            )
        else:
            raster_settings = RasterizationSettings(
                image_size=(module_cfg.img_size[1], module_cfg.img_size[0]),
                blur_radius=0.0,
                # faces_per_pixel=50,
                # max_faces_per_bin=20000,
                bin_size=0,
            )
        self.rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        self.shader = NormalShader(device='cuda')

        self.soft_mask = module_cfg.soft_mask

        self.sigma = 1e-4 if not hasattr(module_cfg, 'sigma') else module_cfg.sigma
        raster_settings_soft = RasterizationSettings(
            image_size=(module_cfg.img_size[1], module_cfg.img_size[0]),
            blur_radius=np.log(1. / 1e-4 - 1.) * self.sigma,
            faces_per_pixel=50,
            bin_size=0,
        )
        self.renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings_soft
            ),
            shader=SoftSilhouetteShader()
        )

    def forward(self, xyzs_observation, vertex_normals, K, E, faces, **kwargs):
        xyzs_ndc = ndc_T_world(xyzs_observation, K, E, self.cfg.img_size[1], self.cfg.img_size[0])

        B, N, _ = xyzs_ndc.shape
        feat = vertex_normals[0]
        mesh = Meshes(xyzs_ndc, faces.unsqueeze(0))

        fragments = self.rasterizer(mesh)
        normal_map = self.shader(fragments, mesh, feat)

        if not self.training:
            return normal_map[..., :-1] * normal_map[..., -1:], None

        mask = self.renderer_silhouette(mesh)[..., -1:]
        return normal_map[..., :-1] * normal_map[..., -1:], mask

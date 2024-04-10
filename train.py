import os
import cv2
import logging
import argparse
import numpy as np
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F

from configs import make_cfg
from dataset.train import Dataset
from models.model import Model

from utils.train_util import cpu_data_to_gpu, make_weights_for_pose_balance
from utils.image_util import to_8b_image
from utils.tb_util import TBLogger
from utils.network_util import mesh_laplacian_smoothing, mesh_color_consistency
from utils.lpips import LPIPS

from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_normal_consistency,
)

from eval import Evaluator

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg",
        default=None,
        type=str
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser.parse_args()


def unpack(rgbs, masks, bgcolors):
    rgbs = rgbs * masks.unsqueeze(-1) + bgcolors[:, None, None, :] * (1 - masks).unsqueeze(-1)
    return rgbs


def evaluate(dataloader, model, tb, split, random_bgcolor=False, n_iters=1e7):
    model.eval()
    evaluator = Evaluator()

    if random_bgcolor:
        bgcolor = torch.tensor(dataloader.dataset.bgcolor).float() / 255.

    for batch_idx, batch in tqdm(enumerate(dataloader)):
        data = cpu_data_to_gpu(
            batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)

        with torch.no_grad():
            pred, mask, _ = model(data['K'], data['E'],
                                  data['cnl_gtfms'], data['dst_Rs'], data['dst_Ts'],
                                  dst_posevec=data['dst_posevec'], i_iter=n_iters)

        truth_imgs = data['target_rgbs'].detach().cpu().numpy()
        if random_bgcolor:
            pred = unpack(pred, mask, bgcolor[None, :].to(pred.device).repeat(pred.shape[0], 1))

        pred_imgs = pred.detach().cpu().numpy()
        for i, (frame_name, pred_img) in enumerate(zip(batch['frame_name'], pred_imgs)):
            pred_img = to_8b_image(pred_img)
            truth_img = to_8b_image(truth_imgs[i])
            evaluator.evaluate(pred_img / 255., truth_img / 255.)

    mse = np.mean(evaluator.mse)
    psnr = np.mean(evaluator.psnr)
    ssim = np.mean(evaluator.ssim)
    lpips = np.mean(evaluator.lpips)
    logging.info(f'evaluate on {split}: mse - {mse:.04f}, psnr - {psnr:.02f}, ssim - {ssim:.04f}, lpips - {lpips:.02f}')

    tb.summ_scalar(f'{split}/mse', mse)
    tb.summ_scalar(f'{split}/psnr', psnr)
    tb.summ_scalar(f'{split}/ssim', ssim)
    tb.summ_scalar(f'{split}/lpips', lpips)

    model.train()


def compute_loss(rgb_pred, mask_pred, outputs, rgb_gt, mask_gt, loss_cfg, data, i_iter, tb=None, **kwargs):
    losses = {}

    loss_rgb = torch.mean(torch.abs(rgb_pred - rgb_gt))
    losses['rgb'] = {
        'unscaled': loss_rgb,
        'scaled': loss_rgb * loss_cfg.rgb.coeff
    }

    loss_mask = torch.mean(torch.abs(mask_pred - mask_gt))
    losses['mask'] = {
        'unscaled': loss_mask,
        'scaled': loss_mask * loss_cfg.mask.coeff
    }

    scale_for_lpips = lambda x: 2 * x - 1
    loss_lpips = torch.mean(kwargs['lpips_func'](
        scale_for_lpips(rgb_pred.permute(0, 3, 1, 2)),
        scale_for_lpips(rgb_gt.permute(0, 3, 1, 2))
    ))
    losses["lpips"] = {
        'unscaled': loss_lpips,
        'scaled': loss_lpips * loss_cfg.lpips.coeff
    }

    if loss_cfg.laplacian.coeff_canonical > 0:
        loss_laplacian_canonical = mesh_laplacian_smoothing(outputs['mesh_canonical'])
        losses['laplacian_canoincal'] = {
            'unscaled': loss_laplacian_canonical,
            'scaled': loss_laplacian_canonical * loss_cfg.laplacian.coeff_canonical
        }

    if loss_cfg.laplacian.coeff_observation > 0:
        loss_laplacian_observation = mesh_laplacian_smoothing(outputs['mesh'])
        losses['laplacian_observation'] = {
            'unscaled': loss_laplacian_observation,
            'scaled': loss_laplacian_observation * loss_cfg.laplacian.coeff_observation
        }

    if loss_cfg.normal.coeff_mask > 0:
        # dilate the mask since the mesh's mask is smaller due to gaussian's volume
        kernel_size = loss_cfg.normal.kernel_size
        mask_gt_dilate = F.max_pool2d(mask_gt.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=kernel_size // 2).squeeze(1)
        mask_gt_normal = mask_gt_dilate
        loss_normal_mask = torch.mean(torch.abs(outputs['normal_mask'] - mask_gt_normal))
        losses['normal_mask'] = {
            'unscaled': loss_normal_mask,
            'scaled': loss_normal_mask * loss_cfg.normal.coeff_mask
        }

    if loss_cfg.normal.coeff_consist > 0:
        loss_normal_consist = mesh_normal_consistency(outputs['mesh'])
        losses['normal_consist'] = {
            'unscaled': loss_normal_consist,
            'scaled': loss_normal_consist * loss_cfg.normal.coeff_consist
        }

    if loss_cfg.color_consist.coeff > 0:
        loss_color_consist = mesh_color_consistency(outputs['colors'], outputs['face_connectivity'])
        losses['color_consist'] = {
            'unscaled': loss_color_consist,
            'scaled': loss_color_consist * loss_cfg.color_consist.coeff
        }

    total_loss = sum([item['scaled'] for item in losses.values()])
    return total_loss, losses


def update_lr(optimizer, iter_step, cfg):
    decay_rate = 0.1
    decay_value = decay_rate ** (iter_step / cfg.lr_decay_steps)
    for param_group in optimizer.param_groups:
        if hasattr(cfg.lr, param_group['name']):
            base_lr = getattr(cfg.lr, param_group['name'])
            new_lrate = base_lr * decay_value
        else:
            new_lrate = cfg.train.lr * decay_value
        param_group['lr'] = new_lrate


def main(args):
    cfg = make_cfg(args.cfg)

    os.makedirs(cfg.save_dir, exist_ok=True)
    # setup logger
    logging_path = os.path.join(cfg.save_dir, 'log.txt')
    logging.basicConfig(
        handlers=[
            logging.FileHandler(logging_path),
            logging.StreamHandler()
        ],
        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    # save config file
    os.makedirs(os.path.join(cfg.save_dir), exist_ok=True)
    with open(os.path.join(cfg.save_dir, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())
    logging.info(f'configs: \n{cfg.dump()}')
    os.makedirs(os.path.join(cfg.save_dir, 'checkpoints'), exist_ok=True)

    # setup tensorboard
    tb_logger = TBLogger(os.path.join(cfg.save_dir, 'tb'), freq=cfg.train.tb_freq)

    # load training data
    train_dataset = Dataset(
        cfg.dataset.train.dataset_path,
        skip=cfg.dataset.train.skip,
        maxframes=cfg.dataset.train.maxframes,
        target_size=cfg.img_size,
        crop_size=cfg.dataset.train.crop_size,
        prefetch=cfg.dataset.train.prefetch,
        split_for_pose=cfg.dataset.train.split_for_pose,
    )
    train_dataloader = torch.utils.data.DataLoader(
        batch_size=cfg.dataset.train.batch_size,
        dataset=train_dataset,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.dataset.train.num_workers)

    # load test data
    if cfg.dataset.test_view.name == 'zju-mocap':
        from dataset.test import Dataset as NovelViewDataset
        test_dataset = NovelViewDataset(
            cfg.dataset.test_view.raw_dataset_path,
            cfg.dataset.test_view.dataset_path,
            test_type='view',
            skip=cfg.dataset.test_view.skip,  # to match monohuman
            exclude_view=cfg.dataset.test_view.exclude_view,
            bgcolor=cfg.bgcolor,
        )
    else:
        from dataset.train import Dataset as NovelViewDataset
        test_dataset = NovelViewDataset(
            cfg.dataset.test_view.dataset_path,
            bgcolor=cfg.bgcolor,
            skip=cfg.dataset.test_view.skip,
            target_size=cfg.model.img_size,
        )
    test_dataloader = torch.utils.data.DataLoader(
        batch_size=cfg.dataset.test_view.batch_size,
        dataset=test_dataset,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.dataset.test_view.num_workers)

    test_on_train_dataset = Dataset(
        cfg.dataset.train.dataset_path,
        skip=30,
        bgcolor=cfg.bgcolor,
        target_size=cfg.model.img_size,
    )
    test_on_train_dataloader = torch.utils.data.DataLoader(
        batch_size=cfg.dataset.test_on_train.batch_size,
        dataset=test_on_train_dataset,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.dataset.test_on_train.num_workers)

    # load model
    model = Model(cfg.model, train_dataset.get_canonical_info())
    model.cuda()
    model.train()

    # load optimizer
    param_groups = model.get_param_groups(cfg.train)
    optimizer = optim.Adam(param_groups, betas=(0.9, 0.999))

    n_iters = 1
    if args.resume:
        ckpt_dir = os.path.join(cfg.save_dir, 'checkpoints')
        max_iter = max([int(filename.split('_')[-1][:-3]) for filename in os.listdir(ckpt_dir)])
        ckpt_path = os.path.join(ckpt_dir, f'iter_{max_iter}.pt')
        ckpt = torch.load(ckpt_path)

        for i in cfg.model.subdivide_iters:
            if max_iter >= i:
                model.subdivide()

        model.load_state_dict(ckpt['network'])
        optimizer.load_state_dict(ckpt['optimizer'])
        logging.info(f"load from checkpoint {ckpt_path}")

        if cfg.train.lr_decay_steps != -1:
            update_lr(optimizer, n_iters, cfg.train)
        n_iters = ckpt['iter'] + 1
        logging.info(f'continue training from iter {n_iters}')
    else:
        ckpt_path = os.path.join(cfg.save_dir, 'checkpoints', f'iter_0.pt')
        torch.save({
            'iter': n_iters,
            'network': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'scheduler': lr_scheduler.state_dict(),
        }, ckpt_path)
        logging.info(f'saved to {ckpt_path}')

    # prepare the loss function
    loss_funcs = {}
    if hasattr(cfg.train.losses, 'lpips'):
        loss_funcs['lpips_func'] = LPIPS(net='vgg')
        for param in loss_funcs['lpips_func'].parameters():
            param.requires_grad = False
        loss_funcs['lpips_func'].cuda()
    if cfg.train.losses.laplacian.coeff_canonical > 0 or cfg.train.losses.laplacian.coeff_observation > 0:
        from utils.network_util import LaplacianLoss
        canonical_info = train_dataset.get_canonical_info()
        loss_funcs['laplacian_func'] = LaplacianLoss(canonical_info['edges'], canonical_info['canonical_vertex'].shape[0]).cuda()

    while n_iters <= cfg.train.total_iters:
        for batch_idx, batch in enumerate(train_dataloader):
            tb_logger.set_global_step(n_iters)

            optimizer.zero_grad()
            data = cpu_data_to_gpu(
                batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)

            rgb, mask, outputs = model(
                data['K'], data['E'],
                data['cnl_gtfms'], data['dst_Rs'], data['dst_Ts'], dst_posevec=data['dst_posevec'],
                canonical_joints=data['dst_tpose_joints'],
                i_iter=n_iters,
                bgcolor=data['bgcolor'],
                tb=tb_logger)

            if cfg.random_bgcolor:
                rgb = unpack(rgb, mask, data['bgcolor'])

            loss, loss_items = compute_loss(
                rgb, mask, outputs,
                data['target_rgbs'], data['target_masks'],
                cfg.train.losses,
                data,
                n_iters,
                tb=tb_logger,
                **loss_funcs
            )

            loss.backward()
            optimizer.step()

            if n_iters in cfg.model.subdivide_iters:
                logging.info(f"subdivide at iter {n_iters}")
                model.subdivide()
                # reinit optimizer
                param_groups = model.get_param_groups(cfg.train)
                optimizer = optim.Adam(param_groups, betas=(0.9, 0.999))

            update_lr(optimizer, n_iters, cfg.train)

            # log to tensorboard
            if n_iters % cfg.train.log_freq == 0:
                tb_logger.summ_scalar('loss_scaled/loss', loss)
                for loss_name, loss_value in loss_items.items():
                    tb_logger.summ_scalar(f'loss_scaled/loss_{loss_name}', loss_value['scaled'])
                    tb_logger.summ_scalar(f'loss_unscaled/loss_{loss_name}', loss_value['unscaled'])

                tb_logger.summ_image('input/rgb', data['target_rgbs'].permute(0, 3, 1, 2)[0])
                tb_logger.summ_image('input/mask', data['target_masks'].unsqueeze(1)[0])
                tb_logger.summ_image('pred/rgb', rgb.permute(0, 3, 1, 2)[0])
                tb_logger.summ_image('pred/mask', mask.unsqueeze(1)[0])

            if n_iters % cfg.train.log_freq == 0:
                loss_str = f"iter {n_iters} - loss: {loss.item():.4f} ("
                for loss_name, loss_value in loss_items.items():
                    loss_str += f"{loss_name}: {loss_value['scaled'].item():.4f}, "
                loss_str = loss_str[:-2] + ")"
                logging.info(loss_str)

            # save
            if n_iters % cfg.train.save_freq == 0:
                ckpt_path = os.path.join(cfg.save_dir, 'checkpoints', f'iter_{n_iters}.pt')
                torch.save({
                    'iter': n_iters,
                    'network': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, ckpt_path)
                logging.info(f'saved to {ckpt_path}')

            # evaluate
            if n_iters % cfg.train.eval_freq == 0:
                evaluate(test_dataloader, model, tb_logger, 'test', cfg.random_bgcolor, n_iters)
                evaluate(test_on_train_dataloader, model, tb_logger, 'train', cfg.random_bgcolor, n_iters)

            n_iters += 1
            if n_iters > cfg.train.total_iters:
                break

    evaluate(test_dataloader, model, tb_logger, 'test', cfg.random_bgcolor, n_iters)


if __name__ == "__main__":
    args = parse_args()
    main(args)

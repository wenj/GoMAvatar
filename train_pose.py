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
import torch.nn as nn

from configs import make_cfg

from dataset.train import Dataset as NovelViewDataset
from models.model import Model

from utils.train_util import cpu_data_to_gpu
from utils.image_util import to_8b_image
from utils.body_util import body_pose_to_body_RTs_tensor
from utils.tb_util import TBLogger
from utils.lpips import LPIPS

from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_normal_consistency,
)

from eval import Evaluator_snapshot as Evaluator

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


def evaluate(cfg, Rhs, Ths, dst_posevecs, dataloader, model, tb, split, random_bgcolor=False, n_iters=1e7):
    save_dir = os.path.join(cfg.save_dir, 'eval', 'test_refine')
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    evaluator = Evaluator()

    if random_bgcolor:
        bgcolor = torch.tensor(dataloader.dataset.bgcolor).float() / 255.

    for batch_idx, batch in tqdm(enumerate(dataloader)):
        data = cpu_data_to_gpu(
            batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)

        with torch.no_grad():
            new_data = update_data(data, dst_posevecs[batch_idx])

            pred, mask, _ = model(
                new_data['K'], new_data['E'],
                new_data['cnl_gtfms'], new_data['dst_Rs'], new_data['dst_Ts'],
                dst_posevec=new_data['dst_posevec'], i_iter=n_iters,
                global_R=Rhs[batch_idx], global_T=Ths[batch_idx])

        truth_imgs = data['target_rgbs'].detach().cpu().numpy()

        if random_bgcolor:
            pred = unpack(pred, mask, bgcolor[None, :].to(pred.device).repeat(pred.shape[0], 1))

        pred_imgs = pred.detach().cpu().numpy()
        for i, (frame_name, pred_img) in enumerate(zip(batch['frame_name'], pred_imgs)):
            pred_img = to_8b_image(pred_img)
            truth_img = to_8b_image(truth_imgs[i])
            evaluator.evaluate(pred_img / 255., truth_img / 255.)

            Image.fromarray(truth_img).save(os.path.join(save_dir, frame_name + '_gt.png'))
            Image.fromarray(pred_img).save(os.path.join(save_dir, frame_name + '.png'))

    psnr = np.mean(evaluator.psnr)
    ssim = np.mean(evaluator.ssim)
    lpips = np.mean(evaluator.lpips)
    logging.info(f'evaluate on {split}: psnr - {psnr:.02f}, ssim - {ssim:.04f}, lpips - {lpips:.04f}')

    if tb is not None:
        tb.summ_scalar(f'{split}/psnr', psnr)
        tb.summ_scalar(f'{split}/ssim', ssim)
        tb.summ_scalar(f'{split}/lpips', lpips)

    model.train()


def update_data(data, dst_posevec):
    new_data = {key: value.detach().clone() for key, value in data.items()}

    dst_tpose_joints = new_data['dst_tpose_joints'][0]
    new_data['dst_Rs'], new_data['dst_Ts'] = body_pose_to_body_RTs_tensor(dst_posevec, dst_tpose_joints)
    new_data['dst_Rs'], new_data['dst_Ts'] = new_data['dst_Rs'][None], new_data['dst_Ts'][None]
    new_data['dst_posevec'] = (dst_posevec[3:] + 1e-2)[None]
    return new_data


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

    if loss_cfg.lpips.coeff > 0:
        scale_for_lpips = lambda x: 2 * x - 1
        loss_lpips = torch.mean(kwargs['lpips_func'](
            scale_for_lpips(rgb_pred.permute(0, 3, 1, 2)),
            scale_for_lpips(rgb_gt.permute(0, 3, 1, 2))
        ))
        losses["lpips"] = {
            'unscaled': loss_lpips,
            'scaled': loss_lpips * loss_cfg.lpips.coeff
        }

    total_loss = sum([item['scaled'] for item in losses.values()])
    return total_loss, losses


def main(args):
    cfg = make_cfg(args.cfg)

    os.makedirs(cfg.save_dir, exist_ok=True)
    # setup logger
    logging_path = os.path.join(cfg.save_dir, 'log_pose.txt')
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

    # load test data of people snapshot
    assert cfg.dataset.test_view.name == 'snapshot'
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

    # load the model
    model = Model(cfg.model, test_dataset.get_canonical_info())
    if len(cfg.model.subdivide_iters) > 0:
        for _ in range(len(cfg.model.subdivide_iters)):
            model.subdivide(need_face_connectivity=False)

    # load checkpoints
    ckpt_dir = os.path.join(cfg.save_dir, 'checkpoints')
    max_iter = max(
        [int(filename.split('_')[-1][:-3]) for filename in os.listdir(ckpt_dir) if 'pose' not in filename])
    ckpt_path = os.path.join(ckpt_dir, f'iter_{max_iter}.pt')
    logging.info(f'loading model from {ckpt_path}')
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['network'], strict=False)

    model.cuda()
    # freeze the network
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # prepare the loss
    loss_funcs = {}
    if hasattr(cfg.train.losses, 'lpips'):
        loss_funcs['lpips_func'] = LPIPS(net='vgg')
        for param in loss_funcs['lpips_func'].parameters():
            param.requires_grad = False
        loss_funcs['lpips_func'].cuda()

    # variables to save results
    Rhs = torch.zeros([len(test_dataset), 3], device='cuda').float()
    Ths = torch.zeros([len(test_dataset), 3], device='cuda').float()
    dst_posevecs = torch.zeros([len(test_dataset), 24 * 3], device='cuda').float()

    dst_posevecs_raw = torch.zeros([len(test_dataset), 24 * 3], device='cuda').float()
    for batch_idx, batch in enumerate(test_dataloader):
        dst_posevecs_raw[batch_idx] = batch['dst_poses'].cuda()

    evaluate(cfg, Rhs, Ths, dst_posevecs_raw, test_dataloader, model, None, 'test', cfg.random_bgcolor, 1e7)

    for batch_idx, batch in enumerate(test_dataloader):
        loss_best = 1e10

        print('batch_idx: ' + str(batch_idx))
        Rh = torch.zeros([3], requires_grad=True, device='cuda')
        Th = torch.zeros([3], requires_grad=True, device='cuda')
        dst_posevec = torch.tensor(batch['dst_poses'][0], requires_grad=True, device='cuda')
        param_groups = [Rh, Th, dst_posevec]

        optimizer = optim.Adam(param_groups, lr=cfg.pose.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.pose.decay, gamma=0.5
        )

        data = cpu_data_to_gpu(batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)
        for _ in range(cfg.pose.iters):
            optimizer.zero_grad()

            new_data = update_data(data, dst_posevec)

            rgb, mask, outputs = model(
                new_data['K'], new_data['E'],
                new_data['cnl_gtfms'], new_data['dst_Rs'], new_data['dst_Ts'],
                dst_posevec=new_data['dst_posevec'],
                canonical_joints=new_data['dst_tpose_joints'],
                global_R=Rh, global_T=Th,
                i_iter=1e7,
                bgcolor=new_data['bgcolor'])

            rgb = unpack(rgb, mask, new_data['bgcolor'])

            loss, loss_items = compute_loss(
                rgb, mask, outputs,
                new_data['target_rgbs'], new_data['target_masks'],
                cfg.train.losses,
                new_data,
                i_iter=1e7,
                tb=None,
                **loss_funcs
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            if loss.item() < loss_best:
                # logging.info('update: {:.4f} -> {:.4f}'.format(loss_best, loss.item()))
                loss_best = loss.item()
                Rhs[batch_idx] = Rh.detach()
                Ths[batch_idx] = Th.detach()
                dst_posevecs[batch_idx] = dst_posevec.detach()

            if _ % 10 == 0:
                loss_str = f"iter {_} - loss: {loss.item():.4f} ("
                for loss_name, loss_value in loss_items.items():
                    loss_str += f"{loss_name}: {loss_value['scaled'].item():.4f}, "
                loss_str = loss_str[:-2] + ")"
                logging.info(loss_str)
        print(Rh, Th)
    evaluate(cfg, torch.zeros_like(Rhs), torch.zeros_like(Ths), dst_posevecs_raw, test_dataloader, model, None, 'test', cfg.random_bgcolor,
             1e7)
    evaluate(cfg, Rhs, Ths, dst_posevecs, test_dataloader, model, None, 'test', cfg.random_bgcolor, 1e7)

    ckpt_path = os.path.join(cfg.save_dir, 'checkpoints', f'pose.pt')
    torch.save({
        'Rhs': Rhs,
        'Ths': Ths,
        'dst_poses': dst_posevecs,
    }, ckpt_path)
    logging.info(f'saved to {ckpt_path}')


if __name__ == "__main__":
    args = parse_args()
    main(args)

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import seaborn as sns

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def grayscale_visualization(im, vmin=None, vmax=None):
    # im should have shape H x W
    im_np = im.detach().cpu().numpy()
    fig = plt.figure()
    plt.imshow(im_np, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('mean value: %.3f' % np.mean(im_np))
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


class TBLogger:
    def __init__(self, log_dir, freq=1, only_scalar=False):
        self.sw = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
        self.freq = freq
        self.only_scalar = only_scalar

    def set_global_step(self, global_step):
        self.global_step = global_step

    def summ_images(self, tag, images, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if not self.only_scalar and global_step % self.freq == 0:
            self.sw.add_images(tag, images.clamp(0, 1), global_step=global_step)

    def summ_image(self, tag, image, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if not self.only_scalar and global_step % self.freq == 0:
            self.sw.add_image(tag, image.clamp(0, 1), global_step=global_step)

    def summ_video(self, tag, video, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if not self.only_scalar and global_step % self.freq == 0:
            self.sw.add_video(tag, video.clamp(0, 1), global_step=global_step)

    def summ_scalar(self, tag, scalar, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if global_step % self.freq == 0:
            self.sw.add_scalar(tag, scalar, global_step=global_step)

    def summ_text(self, tag, text_string, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if global_step % self.freq == 0:
            self.sw.add_text(tag, text_string, global_step=global_step)

    def summ_pts_on_image(self, tag, image, pts, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if global_step % self.freq == 0:
            image = image.detach().cpu().numpy()
            image = image.transpose(1, 2, 0) # H x W x 3
            # use .copy() because of https://stackoverflow.com/questions/23830618/python-opencv-typeerror-layout-of-the-output-array-incompatible-with-cvmat
            image = np.array(image * 255., dtype=np.uint8).copy()
            for pt in pts:
                x, y = int(round(pt[0])), int(round(pt[1]))
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    image = cv2.circle(image, (x, y), radius=1, color=(255, 0, 0), thickness=-1)
            image = image.transpose(2, 0, 1)
            self.sw.add_image(tag, image / 255., global_step=global_step)

    def summ_feat(self, tag, feat, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if global_step % self.freq == 0:
            feat_np = feat.detach().cpu().numpy()
            C, H, W = feat_np.shape
            feat_np = feat_np.transpose(1, 2, 0).reshape(-1, C)

            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            pca.fit(feat_np)
            feat_rgb = pca.transform(feat_np).reshape(H, W, 3).transpose(2, 0, 1)
            feat_min = np.min(feat_rgb)
            feat_max = np.max(feat_rgb)
            feat_rgb = (feat_rgb - feat_min) / (feat_max - feat_min)
            self.sw.add_image(tag, feat_rgb, global_step=global_step)

    def summ_hist(self, tag, values, bins='tensorflow', global_step=None):
        if global_step is None:
            global_step = self.global_step
        if global_step % self.freq == 0:
            self.sw.add_histogram(tag, values.reshape(-1), bins=bins, global_step=global_step)

    def summ_error(self, tag, err, vmin=None, vmax=None, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if not self.only_scalar and global_step % self.freq == 0:
            err_img = grayscale_visualization(err, vmin, vmax)
            self.sw.add_image(tag, err_img.transpose(2, 0, 1), global_step=global_step)

    def summ_graph(self, model, inputs=None):
        self.sw.add_graph(model, inputs)

    def summ_pointcloud(self, tag, pts, colors, radius=None, faces=None, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if global_step % self.freq == 0:
            if colors is not None:
                colors = (colors * 255.).int()
            self.sw.add_mesh(tag, pts, colors=colors, faces=faces, global_step=global_step)

    def summ_pointcloud2d(self, tag, pts, img_size, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if global_step % self.freq == 0:
            W, H = img_size
            img = pts.new_zeros(H, W)
            x = torch.round(pts[0, :, 0]).long()
            y = torch.round(pts[0, :, 1]).long()
            img[y, x] = 1.
            self.sw.add_image(tag, img[None], global_step=global_step)

    def summ_traj_on_image(self, tag, image, pts, global_step=None):
        if global_step is None:
            global_step = self.global_step
        if global_step % self.freq == 0:
            N = pts.shape[0]
            colors = np.array(sns.color_palette("coolwarm", N))
            image = (image.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)
            pts = pts.detach().cpu().numpy()
            for pt, color in zip(pts, colors):
                x, y = pt.astype(int)
                color = (color * 255.).astype(np.uint8).tolist()
                cv2.circle(image, (x, y), 2, color, -1)
            self.sw.add_image(tag, (torch.tensor(image) / 255.).permute(2, 0, 1), global_step=global_step)

    def close(self):
        self.sw.close()
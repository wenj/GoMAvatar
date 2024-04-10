import os
import argparse

import torch

from .yacs import CfgNode as CN


# pylint: disable=redefined-outer-name


def make_cfg(cfg_filename):
	cfg = CN()
	cfg.merge_from_file('configs/default.yaml')
	if cfg_filename is not None:
		cfg.merge_from_file(cfg_filename)

	log_root = 'log' if not hasattr(cfg, 'save_dir') else cfg.save_dir
	cfg.save_dir = os.path.join(log_root, cfg.exp_name)

	return cfg

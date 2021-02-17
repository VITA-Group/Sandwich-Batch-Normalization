# -*- coding: utf-8 -*-
# @Date    : 12/24/19
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import logging
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import make_grid

logger = logging.getLogger(__name__)


def create_logger(args, cfg_name, phase='train', output_dir='output', log_dir='log', test=False):
    root_output_dir = Path(output_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = args.dataset
    dataset = dataset.replace(':', '_')
    model = args.model + "-" + args.norm_module

    final_output_dir = root_output_dir / dataset / model / phase / cfg_name

    if not final_output_dir.exists():
        print('=> creating {}'.format(final_output_dir))
        final_output_dir.mkdir(parents=True, exist_ok=False)
    else:
        print(f'=> resuming with {final_output_dir}')

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    if test:
        log_file = '{}_{}_{}.log'.format(cfg_name, time_str, 'test')
        tensorboard_log_dir = Path(log_dir) / dataset / model / 'test' / \
                              (cfg_name + '_' + time_str)
    else:
        log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
        tensorboard_log_dir = Path(log_dir) / dataset / model / phase / \
                              (cfg_name + '_' + time_str)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(args.log_level)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))


def load_params(param, new_param):
    for p, new_p in zip(param, new_param):
        p.data.copy_(new_p)


def copy_params(param):
    flatten = deepcopy(list(p.data for p in param))
    return flatten


@torch.no_grad()
def log_image(fixed_z, gen_net, writer_dict, epoch, fixed_pseudo_label=None):
    writer = writer_dict['writer']
    gen_net = gen_net.eval()

    # generate images
    if fixed_pseudo_label is None:
        sample_imgs = gen_net(fixed_z)
    else:
        sample_imgs = gen_net(fixed_z, fixed_pseudo_label)
    img_grid = make_grid(sample_imgs, nrow=int(np.ceil(np.sqrt(fixed_z.shape[0]))), normalize=True, scale_each=True)
    writer.add_image('sampled_images', img_grid, epoch)

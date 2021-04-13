# -*- coding: utf-8 -*-
# @Date    : 2/16/21
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import os

import torch

import models
from cfg import parse_args
from functions import validate
from metrics.inception_score import init_inception
from utils import create_logger

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def get_network_func(args):
    """
    Retrieves the transformation module by name.
    """
    if args.dataset == "cifar10":
        networks = {
            "sngan": (models.SnganGen32(args), models.SnganDis32(args)),
            "autogan": (
                models.AutoganGen32(args),
                models.SnganDis32(args),
            ),  # using sngan's dis_net
        }
    elif args.dataset == "imagenet":
        networks = {
            "sngan": (models.SnganGen128(args), models.SnganDis128(args)),
        }
    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}.")

    assert args.model in networks.keys(), "Networks function '{}' not supported".format(
        args.model
    )
    return networks[args.model]


def main(args):
    assert args.exp_name
    logger, final_output_dir, _ = create_logger(args, args.exp_name, "test")
    args.sample_path = final_output_dir
    torch.cuda.manual_seed(args.random_seed)

    # set tf env
    init_inception()

    # fid stat
    if args.dataset.lower() == "cifar10":
        args.n_classes = 10
        fid_stat = "fid_stat/fid_stats_cifar10_train.npz"
    elif args.dataset.lower() == "imagenet":
        # TODO: Support intra-FID
        args.n_classes = 143
        fid_stat = None
    else:
        raise NotImplementedError(f"no fid stat for {args.dataset.lower()}")
    if fid_stat:
        assert os.path.exists(fid_stat), f"{fid_stat} not found"

    # get network
    gen_net, _ = get_network_func(args)
    gen_net.cuda()

    # load checkpoint
    checkpoint_file = args.load_path
    assert os.path.exists(checkpoint_file), print(
        f"checkpoint file {checkpoint_file} not found."
    )
    logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
    gen_net.load_state_dict(checkpoint)
    logger.info(f"=> loaded checkpoint '{checkpoint_file}' ")

    # evaluation
    torch.cuda.empty_cache()
    inception_score, fid_score = validate(args, fid_stat, gen_net, None)
    logger.info(f"Inception score: {inception_score}, FID score: {fid_score} ")


if __name__ == "__main__":
    config = parse_args()
    main(config)

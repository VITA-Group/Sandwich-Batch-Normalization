# -*- coding: utf-8 -*-
# @Date    : 2/16/21
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import os
import pprint
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

import models
from cfg import parse_args
from datasets import ImageDataset
from functions import train_conditional, validate
from metrics.inception_score import init_inception
from utils import copy_params, create_logger, load_params, log_image, save_checkpoint

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
    logger, final_output_dir, tb_log_dir = create_logger(args, args.exp_name, "train")
    args.sample_path = final_output_dir
    torch.cuda.manual_seed(args.random_seed)

    # set tf env
    init_inception()

    # fid stat
    if args.dataset.lower() == "cifar10":
        fid_stat = "fid_stat/fid_stats_cifar10_train.npz"
    elif args.dataset.lower() == "imagenet":
        fid_stat = None
    else:
        raise NotImplementedError(f"no fid stat for {args.dataset.lower()}")
    if fid_stat:
        assert os.path.exists(fid_stat), f"Cannot find {fid_stat}."

    # set up data loader
    dataset = ImageDataset(args)
    train_loader = dataset.train

    # epoch number for dis_net
    args.max_epoch = args.max_epoch * args.n_critic
    if args.max_iter:
        args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loader))

    # get network
    gen_net, dis_net = get_network_func(args)
    gen_net.cuda()
    dis_net.cuda()

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            if m.weight is not None:
                if args.init_type == "normal":
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
                elif args.init_type == "orth":
                    nn.init.orthogonal_(m.weight.data)
                elif args.init_type == "xavier_uniform":
                    nn.init.xavier_uniform(m.weight.data, 1.0)
                else:
                    raise NotImplementedError(
                        "{} unknown inital type".format(args.init_type)
                    )
        elif classname.find("BatchNorm2d") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    gen_net.apply(weights_init)
    dis_net.apply(weights_init)

    # set optimizer
    gen_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gen_net.parameters()),
        args.g_lr,
        (args.beta1, args.beta2),
    )
    dis_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, dis_net.parameters()),
        args.d_lr,
        (args.beta1, args.beta2),
    )

    start_epoch = 0
    best_fid = 1e4
    gen_avg_param = copy_params(gen_net.parameters())
    checkpoint_file = os.path.join(final_output_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(
            checkpoint_file, map_location=lambda storage, loc: storage
        )
        start_epoch = checkpoint["epoch"]
        best_fid = checkpoint["best_fid"]
        gen_net.load_state_dict(checkpoint["gen_state_dict"])
        dis_net.load_state_dict(checkpoint["dis_state_dict"])
        gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])
        dis_optimizer.load_state_dict(checkpoint["dis_optimizer"])
        avg_gen_net = deepcopy(gen_net)
        avg_gen_net.load_state_dict(checkpoint["avg_gen_state_dict"])
        gen_avg_param = copy_params(avg_gen_net.parameters())
        del avg_gen_net
        logger.info(
            "=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint["epoch"]
            )
        )

    logger.info(pprint.pformat(args))
    writer_dict = {
        "writer": SummaryWriter(tb_log_dir),
        "train_global_steps": start_epoch * len(train_loader),
        "valid_global_steps": start_epoch // args.val_freq,
    }

    # For visualization
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))
    if args.n_classes > 0:
        fixed_pseudo_label = torch.randint(
            low=0, high=args.n_classes, size=(25,), device="cuda"
        )
    else:
        fixed_pseudo_label = None

    # train loop
    for epoch in tqdm(
        range(int(start_epoch), int(args.max_epoch)), desc="total progress"
    ):
        lr_schedulers = None
        train_conditional(
            args,
            gen_net,
            dis_net,
            gen_optimizer,
            dis_optimizer,
            gen_avg_param,
            train_loader,
            epoch,
            writer_dict,
            lr_schedulers,
        )

        # log image
        log_image(fixed_z, gen_net, writer_dict, epoch, fixed_pseudo_label)

        if epoch % args.val_freq == 0 or epoch == int(args.max_epoch) - 1:
            backup_param = copy_params(gen_net.parameters())
            load_params(gen_net.parameters(), gen_avg_param)
            torch.cuda.empty_cache()
            inception_score, fid_score = validate(args, fid_stat, gen_net, writer_dict)
            logger.info(
                f"Inception score: {inception_score}, FID score: {fid_score} || @ epoch {epoch}."
            )
            load_params(gen_net.parameters(), backup_param)
            if fid_score < best_fid:
                best_fid = fid_score
                is_best = True
                logger.info(
                    f"Best performance w.r.t FID: FID score: {fid_score} || @ epoch {epoch}."
                )
            else:
                is_best = False
        else:
            is_best = False

        avg_gen_net = deepcopy(gen_net)
        load_params(avg_gen_net.parameters(), gen_avg_param)
        states = {
            "epoch": epoch + 1,
            "model": args.model,
            "gen_state_dict": gen_net.state_dict(),
            "dis_state_dict": dis_net.state_dict(),
            "avg_gen_state_dict": avg_gen_net.state_dict(),
            "gen_optimizer": gen_optimizer.state_dict(),
            "dis_optimizer": dis_optimizer.state_dict(),
            "best_fid": best_fid,
        }
        save_checkpoint(states, is_best, final_output_dir)
        logger.info(f"=> latest checkpoint saved to {final_output_dir}")
        if (
            args.snapshot
            and epoch % args.snapshot == 0
            or epoch == int(args.max_epoch) - 1
        ):
            save_checkpoint(
                states, False, final_output_dir, f"checkpoint_epoch{epoch + 1}.pth"
            )
            logger.info(f"=> snapshot checkpoint saved")
        del avg_gen_net


if __name__ == "__main__":
    config = parse_args()
    main(config)

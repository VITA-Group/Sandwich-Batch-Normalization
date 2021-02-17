# -*- coding: utf-8 -*-
# @Date    : 2/17/21
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import logging
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from tqdm import tqdm

from metrics import get_inception_score, calculate_fid_given_paths_torch

logger = logging.getLogger(__name__)


def train_conditional(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param,
                      train_loader, epoch, writer_dict, image_counter=None, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()

    for iter_idx, (imgs, true_label) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']
        if image_counter is not None:
            image_counter.step(imgs.shape[0])

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)
        true_label = true_label.type(torch.cuda.LongTensor)

        if iter_idx == 0:
            # log real image
            img_grid = make_grid(real_imgs[:25], nrow=5, normalize=True, scale_each=True)
            writer.add_image('real_images', img_grid, epoch)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs, true_label)

        pseudo_label = torch.randint(low=0, high=args.n_classes, size=(z.shape[0],), device='cuda')
        fake_imgs = gen_net(z, pseudo_label).detach()
        assert fake_imgs.size() == real_imgs.size()
        fake_validity = dis_net(fake_imgs, pseudo_label)

        # cal loss
        if args.loss == 'hinge':
            d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                     torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        else:
            raise NotImplementedError(args.loss)
        d_loss.backward()
        dis_optimizer.step()

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            pseudo_label = torch.randint(low=0, high=args.n_classes, size=(gen_z.shape[0],), device='cuda')
            gen_imgs = gen_net(gen_z, pseudo_label)
            fake_validity = dis_net(gen_imgs, pseudo_label)

            # cal loss
            if args.loss == 'hinge' or args.loss == 'wgangp':
                g_loss = -torch.mean(fake_validity)
            else:
                raise NotImplementedError(args.loss)
            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))
            writer.add_scalar('d_loss', d_loss.item(), global_steps)
            writer.add_scalar('g_loss', g_loss.item(), global_steps)

        writer_dict['train_global_steps'] = global_steps + 1


@torch.no_grad()
def validate(args, fid_stat, gen_net: nn.Module, writer_dict=None):
    """
    Compute both IS and FID (torch).
    :param args:
    :param fid_stat:
    :param gen_net:
    :param writer_dict:
    :return:
    """
    # eval mode
    gen_net = gen_net.eval()

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    normal_img_list = []
    gen_img_list = []
    for _ in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        if args.n_classes > 0:
            random_label = torch.randint(low=0, high=args.n_classes, size=(z.shape[0],), device='cuda')
            gen_imgs = gen_net(z, random_label)
        else:
            gen_imgs = gen_net(z)
        if fid_stat is not None:
            gen_img_list += [deepcopy(gen_imgs)]

        nomal_imgs = gen_imgs.mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
                                                                                                torch.uint8).numpy()
        normal_img_list.extend(list(nomal_imgs))

    # get inception score
    logger.info('=> calculate inception score')
    mean, std = get_inception_score(normal_img_list)
    print(f"Inception score: {mean}")
    del normal_img_list

    # get fid score
    if fid_stat is not None:
        logger.info('=> calculate FID')
        fid = calculate_fid_given_paths_torch(torch.cat(gen_img_list, 0), fid_stat)
        print(f"FID: {fid}")
    else:
        logger.info('=> skip calculate FID')
        fid = 1e4
    del gen_img_list
    torch.cuda.empty_cache()

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('Inception_score/mean', mean, global_steps)
        writer.add_scalar('Inception_score/std', std, global_steps)
        writer.add_scalar('FID', fid, global_steps)

        writer_dict['valid_global_steps'] = global_steps + 1

    return mean, fid

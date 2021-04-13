# -*- coding: utf-8 -*-
# @Date    : 2/16/21
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import torch
import torch.nn as nn

from .modules import GenBlock, OptimizedDisBlock, DisBlock

__all__ = ["Generator", "Discriminator"]


class Generator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Generator, self).__init__()
        self.bottom_width = args.bottom_width
        self.activation = activation
        self.n_classes = args.n_classes
        self.ch = args.gf_dim
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.ch)
        self.block2 = GenBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            upsample=True,
            n_classes=args.n_classes,
        )
        self.block3 = GenBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            upsample=True,
            n_classes=args.n_classes,
        )
        self.block4 = GenBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            upsample=True,
            n_classes=args.n_classes,
        )
        self.b5 = nn.BatchNorm2d(self.ch)
        self.c5 = nn.Conv2d(self.ch, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z, y=None):
        if y is not None:
            assert isinstance(y, torch.Tensor)
        h = z
        h = self.l1(h).view(-1, self.ch, self.bottom_width, self.bottom_width)
        for blk_idx in range(2, 5):
            h = getattr(self, f"block{blk_idx}")(h, y)
        h = self.b5(h)
        h = self.activation(h)
        h = nn.Tanh()(self.c5(h))
        return h


class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisBlock(
            args, self.ch, self.ch, activation=activation, downsample=True
        )
        self.block3 = DisBlock(
            args, self.ch, self.ch, activation=activation, downsample=False
        )
        self.block4 = DisBlock(
            args, self.ch, self.ch, activation=activation, downsample=False
        )
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if args.n_classes > 0:
            self.l_y = nn.Embedding(args.n_classes, self.ch)
        if args.d_sn:
            self.l5 = nn.utils.spectral_norm(self.l5)
            if args.n_classes > 0:
                self.l_y = nn.utils.spectral_norm(self.l_y)

    def forward(self, x, y=None):
        if y is not None:
            assert isinstance(y, torch.Tensor)
        h = x
        for blk_idx in range(1, 5):
            h = getattr(self, f"block{blk_idx}")(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output

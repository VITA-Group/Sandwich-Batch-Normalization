# -*- coding: utf-8 -*-
# @Date    : 2/16/21
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import torch
import torch.nn as nn

from .modules import DisBlock, GenBlock, OptimizedDisBlock


class Generator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Generator, self).__init__()
        self.bottom_width = args.bottom_width
        self.activation = activation
        self.ch = args.gf_dim
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.ch * 8)
        self.block2 = GenBlock(
            args,
            self.ch * 8,
            self.ch * 8,
            activation=activation,
            upsample=True,
            n_classes=args.n_classes,
        )
        self.block3 = GenBlock(
            args,
            self.ch * 8,
            self.ch * 4,
            activation=activation,
            upsample=True,
            n_classes=args.n_classes,
        )
        self.block4 = GenBlock(
            args,
            self.ch * 4,
            self.ch * 4,
            activation=activation,
            upsample=True,
            n_classes=args.n_classes,
        )
        self.block5 = GenBlock(
            args,
            self.ch * 4,
            self.ch * 2,
            activation=activation,
            upsample=True,
            n_classes=args.n_classes,
        )
        self.block6 = GenBlock(
            args,
            self.ch * 2,
            self.ch,
            activation=activation,
            upsample=True,
            n_classes=args.n_classes,
        )
        self.b7 = nn.BatchNorm2d(self.ch)
        self.c7 = nn.Conv2d(self.ch, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z, y=None):
        if y is not None:
            assert isinstance(y, torch.Tensor)
        h = z
        h = self.l1(h).view(-1, self.ch * 8, self.bottom_width, self.bottom_width)
        for block_idx in range(2, 7):
            h = getattr(self, f"block{block_idx}")(h, y)
        h = self.b7(h)
        h = self.activation(h)
        h = torch.tanh(self.c7(h))
        return h


class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisBlock(
            args, self.ch, self.ch * 2, activation=activation, downsample=True
        )
        self.block3 = DisBlock(
            args, self.ch * 2, self.ch * 4, activation=activation, downsample=True
        )
        self.block4 = DisBlock(
            args, self.ch * 4, self.ch * 8, activation=activation, downsample=True
        )
        self.block5 = DisBlock(
            args, self.ch * 8, self.ch * 8, activation=activation, downsample=True
        )
        self.block6 = DisBlock(
            args, self.ch * 8, self.ch * 8, activation=activation, downsample=False
        )
        self.l7 = nn.Linear(self.ch * 8, 1, bias=False)
        if args.n_classes > 0:
            self.l_y = nn.Embedding(args.n_classes, self.ch * 8)

        if args.d_sn:
            self.l7 = nn.utils.spectral_norm(self.l7)
            if args.n_classes > 0:
                self.l_y = nn.utils.spectral_norm(self.l_y)

    def forward(self, x, y=None):
        if y is not None:
            assert isinstance(y, torch.Tensor)
        h = x
        for block_idx in range(1, 7):
            h = getattr(self, f"block{block_idx}")(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l7(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output

# -*- coding: utf-8 -*-
# @Date    : 2/17/21
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

from torch import nn
from .modules import Cell


class Generator(nn.Module):
    """
    Conditional norm version autogan
    """

    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.gf_dim
        self.bottom_width = args.bottom_width
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * args.gf_dim)
        self.cell1 = Cell(
            args,
            args.gf_dim,
            args.gf_dim,
            "nearest",
            num_skip_in=0,
            short_cut=True,
            norm="bn",
            n_classes=args.n_classes,
        )
        self.cell2 = Cell(
            args,
            args.gf_dim,
            args.gf_dim,
            "bilinear",
            num_skip_in=1,
            short_cut=True,
            norm="bn",
            n_classes=args.n_classes,
        )
        self.cell3 = Cell(
            args,
            args.gf_dim,
            args.gf_dim,
            "nearest",
            num_skip_in=2,
            short_cut=False,
            norm="bn",
            n_classes=args.n_classes,
        )
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(args.gf_dim),
            nn.ReLU(),
            nn.Conv2d(args.gf_dim, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, y=None):
        h = self.l1(z).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h1_skip_out, h1 = self.cell1(h, y=y)
        h2_skip_out, h2 = self.cell2(h1, (h1_skip_out,), y=y)
        _, h3 = self.cell3(h2, (h1_skip_out, h2_skip_out), y=y)
        output = self.to_rgb(h3)
        return output

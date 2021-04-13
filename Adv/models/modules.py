# -*- coding: utf-8 -*-
# @Date    : 2/16/21
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxBN(nn.Module):
    def __init__(self, num_features, num_bns=2):
        super().__init__()
        self.num_features = num_features
        self.bn_list = nn.ModuleList(
            nn.BatchNorm2d(num_features, affine=True) for _ in range(num_bns)
        )

    def forward(self, x, y):
        out = self.bn_list[y](x)
        return out


class SaAuxBN(nn.Module):
    def __init__(self, num_features, num_bns=2):
        super().__init__()
        self.num_features = num_features
        self.num_bns = num_bns
        self.bn_list = nn.ModuleList(
            nn.BatchNorm2d(num_features, affine=False) for _ in range(num_bns)
        )
        self.embed = nn.Embedding(num_bns + 1, num_features * 2)

        self.embed.weight.data[:, :num_features].normal_(
            1, 0.02
        )  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn_list[y](x)
        gamma_shared, beta_shared = self.embed(
            self.num_bns * torch.ones(x.size(0)).long().cuda()
        ).chunk(2, 1)
        out = gamma_shared.view(-1, self.num_features, 1, 1) * out + beta_shared.view(
            -1, self.num_features, 1, 1
        )
        gamma, beta = self.embed(y * torch.ones(x.size(0)).long().cuda()).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
            -1, self.num_features, 1, 1
        )
        return out


class BaseBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, norm_module, stride=1):
        super().__init__()
        self.bn1 = norm_module(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = norm_module(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"forward function is not implemented")


class NormalBasicBlock(BaseBlock):
    """
    Pre-activation version of the BasicBlock.
    """

    def __init__(self, in_planes, planes, norm_module=nn.BatchNorm2d, stride=1):
        super().__init__(in_planes, planes, norm_module, stride)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class AuxBasicBlock(BaseBlock):
    """
    Pre-activation version of the BasicBlock. It has two separate bn conditions
    """

    def __init__(self, in_planes, planes, norm_module, stride=1):
        super().__init__(in_planes, planes, norm_module, stride)

    def forward(self, inputs):
        assert len(inputs) == 2
        x = inputs[0]
        y = inputs[1]

        out = F.relu(self.bn1(x, y=y))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out, y=y)))
        out += shortcut

        return [out, y]

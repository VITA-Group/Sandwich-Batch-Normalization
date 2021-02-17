# -*- coding: utf-8 -*-
# @Date    : 2/16/21
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import torch.nn as nn
import torch.nn.functional as F
from .modules import NormalBasicBlock, AuxBasicBlock, AuxBN, SaAuxBN
from .utils import NormalizeByChannelMeanStd


def get_network_func(name):
    """
    Retrieves the transformation module by name.
    """
    networks = {
        "bn": resnet18(),
        "auxbn": aux_resnet18(),
        "saauxbn": saaux_resnet18(),
    }
    assert (
            name in networks.keys()
    ), "Networks function '{}' not supported".format(name)
    return networks[name]


class BaseResNet(nn.Module):
    def __init__(self, block, norm_module, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.normal = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, norm_module, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, norm_module, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, norm_module, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, norm_module, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, norm_module, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, norm_module, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"forward function is not implemented")


class NormalResNet(BaseResNet):
    def __init__(self, block, norm_module, num_blocks, num_classes=10):
        super().__init__(block, norm_module, num_blocks, num_classes)

    def forward(self, x, flag):
        x = self.normal(x)
        # out = F.relu(self.bn(self.conv1(x)))
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class AuxResNet(BaseResNet):
    def __init__(self, block, norm_module, num_blocks, num_classes=10):
        super().__init__(block, norm_module, num_blocks, num_classes)

    def forward(self, x, flag):
        x = self.normal(x)
        out = self.conv1(x)
        out = self.layer1([out, flag])[0]
        out = self.layer2([out, flag])[0]
        out = self.layer3([out, flag])[0]
        out = self.layer4([out, flag])[0]
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def aux_resnet18():
    return AuxResNet(AuxBasicBlock, AuxBN, [2, 2, 2, 2])


def saaux_resnet18():
    return AuxResNet(AuxBasicBlock, SaAuxBN, [2, 2, 2, 2])


def resnet18():
    return NormalResNet(NormalBasicBlock, nn.BatchNorm2d, [2, 2, 2, 2])

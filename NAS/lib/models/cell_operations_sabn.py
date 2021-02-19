##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import torch
import torch.nn as nn

from .cell_operations import ResNetBasicblock, Zero, Identity
from .norm_modules import SandwichBatchNorm2d

__all__ = ['OPS', 'ResNetBasicblock']

OPS = {
    'none': lambda C_in, C_out, stride, affine, track_running_stats, num_ops, num_prev_nodes, is_node_zero: Zero(C_in,
                                                                                                                 C_out,
                                                                                                                 stride),
    'avg_pool_3x3': lambda C_in, C_out, stride, affine, track_running_stats, num_ops, num_prev_nodes,
                           is_node_zero: POOLING(C_in, C_out, stride, 'avg', affine, track_running_stats, num_ops,
                                                 num_prev_nodes, is_node_zero),
    'nor_conv_3x3': lambda C_in, C_out, stride, affine, track_running_stats, num_ops, num_prev_nodes,
                           is_node_zero: ReLUConvBN(C_in, C_out, (3, 3), (stride, stride), (1, 1), (1, 1), affine,
                                                    track_running_stats, num_ops, num_prev_nodes, is_node_zero),
    'nor_conv_1x1': lambda C_in, C_out, stride, affine, track_running_stats, num_ops, num_prev_nodes,
                           is_node_zero: ReLUConvBN(C_in, C_out, (1, 1), (stride, stride), (0, 0), (1, 1), affine,
                                                    track_running_stats, num_ops, num_prev_nodes, is_node_zero),
    'skip_connect': lambda C_in, C_out, stride, affine, track_running_stats, num_ops, num_prev_nodes,
                           is_node_zero: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out,
                                                                                                           stride,
                                                                                                           affine,
                                                                                                           track_running_stats,
                                                                                                           num_ops,
                                                                                                           num_prev_nodes,
                                                                                                           is_node_zero),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True,
                 num_prev_ops=1, num_prev_nodes=1, is_node_zero=False):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
        )
        self.bn = SandwichBatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats,
                                      num_prev_ops=num_prev_ops, num_prev_nodes=num_prev_nodes,
                                      is_node_zero=is_node_zero)

    def forward(self, x, prev_op_idx=[0], first_layer=False):
        x = self.op(x)
        output = self.bn(x, prev_op_idx, first_layer)
        return output


class POOLING(nn.Module):

    def __init__(self, C_in, C_out, stride, mode, affine=True, track_running_stats=True, num_prev_ops=1,
                 num_prev_nodes=1, is_node_zero=False):
        super(POOLING, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 1, affine, track_running_stats,
                                         num_prev_ops=num_prev_ops, num_prev_nodes=num_prev_nodes,
                                         is_node_zero=is_node_zero)
        if mode == 'avg':
            self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == 'max':
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError('Invalid mode={:} in POOLING'.format(mode))

    def forward(self, inputs, prev_op_idx=[0], first_layer=False):
        if self.preprocess:
            x = self.preprocess(inputs, prev_op_idx, first_layer)
        else:
            x = inputs
        return self.op(x)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, stride, affine, track_running_stats, num_prev_ops=1, num_prev_nodes=1,
                 is_node_zero=False):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False))
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        else:
            raise ValueError('Invalid stride : {:}'.format(stride))
        self.bn = SandwichBatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats,
                                      num_prev_ops=num_prev_ops,
                                      num_prev_nodes=num_prev_nodes, is_node_zero=is_node_zero)

    def forward(self, x, prev_op_idx=[0], first_layer=False):
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        out = self.bn(out, prev_op_idx, first_layer)
        return out

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)

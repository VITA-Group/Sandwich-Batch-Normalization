# -*- coding: utf-8 -*-
# @Date    : 2/16/21
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import torch
import torch.nn as nn
import torch.nn.functional as F


class SandwichBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=True)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class CategoricalConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class Cell(nn.Module):
    """
        Cell of AutoGAN
    """
    def __init__(self, args, in_channels, out_channels, up_mode, ksize=3, num_skip_in=0, short_cut=False, norm=None,
                 n_classes=0):
        super(Cell, self).__init__()
        UP_MODES = ['nearest', 'bilinear']
        NORMS = ['in', 'bn']
        self.c1 = nn.Conv2d(in_channels, out_channels, ksize, padding=ksize//2)
        self.c2 = nn.Conv2d(out_channels, out_channels, ksize, padding=ksize//2)
        assert up_mode in UP_MODES
        self.up_mode = up_mode
        self.norm = norm
        self.n_classes = n_classes
        if norm:
            if n_classes > 0:
                if args.norm_module.lower() == "sabn":
                    self.n1 = SandwichBatchNorm2d(in_channels, n_classes)
                    self.n2 = SandwichBatchNorm2d(out_channels, n_classes)
                elif args.norm_module.lower() == "ccbn":
                    self.n1 = CategoricalConditionalBatchNorm2d(in_channels, n_classes)
                    self.n2 = CategoricalConditionalBatchNorm2d(out_channels, n_classes)
                else:
                    raise NotImplementedError(f"Unknown norm module {args.norm_module} for conditional generation.")
            else:
                assert norm in NORMS
                if norm == 'bn':
                    self.n1 = nn.BatchNorm2d(in_channels)
                    self.n2 = nn.BatchNorm2d(out_channels)
                elif norm == 'in':
                    self.n1 = nn.InstanceNorm2d(in_channels)
                    self.n2 = nn.InstanceNorm2d(out_channels)
                else:
                    raise NotImplementedError(f"Unknown norm module {norm} for unconditional generation.")

        # inner shortcut
        self.c_sc = None
        if short_cut:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # cross scale skip
        self.skip_in_ops = None
        if num_skip_in:
            self.skip_in_ops = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=1) for _ in range(num_skip_in)])

    def forward(self, x, skip_ft=None, y=None):
        residual = x

        # first conv
        if self.norm:
            residual = self.n1(residual, y) if y is not None else self.n1(residual)
        h = nn.ReLU()(residual)
        h = F.interpolate(h, scale_factor=2, mode=self.up_mode)
        _, _, ht, wt = h.size()
        h = self.c1(h)
        h_skip_out = h

        # second conv
        if self.skip_in_ops:
            assert len(self.skip_in_ops) == len(skip_ft)
            for ft, skip_in_op in zip(skip_ft, self.skip_in_ops):
                h += skip_in_op(F.interpolate(ft, size=(ht, wt), mode=self.up_mode))
        if self.norm:
            h = self.n2(h, y) if y is not None else self.n2(h)
        h = nn.ReLU()(h)
        final_out = self.c2(h)

        # shortcut
        if self.c_sc:
            final_out += self.c_sc(F.interpolate(x, scale_factor=2, mode=self.up_mode))

        return h_skip_out, final_out


class GenBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.ReLU(), upsample=False, n_classes=0):
        super().__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)

        if n_classes > 0:
            # conditional
            if args.norm_module.lower() == "sabn":
                self.b1 = SandwichBatchNorm2d(in_channels, n_classes)
                self.b2 = SandwichBatchNorm2d(hidden_channels, n_classes)
            elif args.norm_module.lower() == "ccbn":
                self.b1 = CategoricalConditionalBatchNorm2d(in_channels, n_classes)
                self.b2 = CategoricalConditionalBatchNorm2d(hidden_channels, n_classes)
            else:
                raise NotImplementedError(f"Unknown norm module {args.norm_module} for conditional generation.")
        else:
            # unconditional
            self.b1 = nn.BatchNorm2d(in_channels)
            self.b2 = nn.BatchNorm2d(hidden_channels)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def upsample_conv(self, x, conv):
        # return conv(nn.UpsamplingNearest2d(scale_factor=2)(x))
        return conv(F.interpolate(x, scale_factor=2))

    def residual(self, x, y=None):
        h = x
        h = self.b1(h, y) if y is not None else self.b1(h)
        h = self.activation(h)
        h = self.upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, y) if y is not None else self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x, y=None):
        return self.residual(x, y) + self.shortcut(x)


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class OptimizedDisBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, ksize=3, pad=1, activation=nn.ReLU()):
        super().__init__()
        self.activation = activation

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        if args.d_sn:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.ReLU(), downsample=False):
        super().__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        if args.d_sn:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if args.d_sn:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)



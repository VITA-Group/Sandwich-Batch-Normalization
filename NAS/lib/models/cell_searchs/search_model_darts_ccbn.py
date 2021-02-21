##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
########################################################
# DARTS: Differentiable Architecture Search, ICLR 2019 #
########################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from ..cell_operations_ccbn import ResNetBasicblock
from .search_cells_ccbn import NAS201SearchCell as SearchCell
from .genotypes import Structure
from ..model_utils import cal_entropy


class TinyNetworkDartsCCBN(nn.Module):

    def __init__(self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats):
        super(TinyNetworkDartsCCBN, self).__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C))

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(
                        num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch_parameters = nn.Parameter(1e-3 * torch.randn(num_edge, len(search_space)))

    @property
    def entropy(self):
        return cal_entropy(self.arch_parameters, -1).mean().item()

    def get_weights(self):
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(self.global_pooling.parameters())
        xlist += list(self.classifier.parameters())
        return xlist

    def get_alphas(self):
        return [self.arch_parameters]

    def _save_arch_parameters(self):
        self._saved_arch_parameters = self.arch_parameters.clone()

    def softmax_arch_parameters(self):
        self._save_arch_parameters()
        self.arch_parameters.data.copy_(F.softmax(self.arch_parameters, dim=-1))

    def restore_arch_parameters(self):
        self.arch_parameters.data.copy_(self._saved_arch_parameters)
        del self._saved_arch_parameters

    def clip(self):
        for line in self.arch_parameters:
            max_index = line.argmax()
            line.data.clamp_(0, 1)
            if line.sum() == 0.0:
                line.data[max_index] = 1.0
            line.data.div_(line.sum())

    def show_alphas(self):
        with torch.no_grad():
            return 'arch-parameters :\n{:}'.format(nn.functional.softmax(self.arch_parameters, dim=-1).cpu())

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return ('{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__,
                                                                                        **self.__dict__))

    def genotype(self):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    weights = self.arch_parameters[self.edge2index[node_str]]
                    op_name = self.op_names[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def forward(self, inputs):
        alphas = nn.functional.softmax(self.arch_parameters, dim=-1)

        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                feature = cell(feature, alphas, i == 0)
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits

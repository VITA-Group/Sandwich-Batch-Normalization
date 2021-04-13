##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import torch.nn as nn
from copy import deepcopy
from ..cell_operations_ccbn import OPS


# This module is used for NAS-Bench-201, represents a small search space with a complete DAG
class NAS201SearchCell(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        stride,
        max_nodes,
        op_names,
        affine=False,
        track_running_stats=True,
    ):
        super().__init__()
        self.num_ops = len(op_names)
        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                if j == 0:
                    xlists = [
                        OPS[op_name](
                            C_in,
                            C_out,
                            stride,
                            affine,
                            track_running_stats,
                            self.num_ops,
                            j,
                            False,
                        )
                        for op_name in op_names
                    ]
                else:
                    xlists = [
                        OPS[op_name](
                            C_in,
                            C_out,
                            1,
                            affine,
                            track_running_stats,
                            self.num_ops,
                            j,
                            False,
                        )
                        for op_name in op_names
                    ]
                self.edges[node_str] = nn.ModuleList(xlists)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def extra_repr(self):
        string = "info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}".format(
            **self.__dict__
        )
        return string

    def forward(self, inputs, weightss, first_layer):
        nodes_accept_op_idx = [[0]]
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            new_nodes_accept_op_idx = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = weightss[self.edge2index[node_str]]
                # argmaxs  = torch.argmax(weights)
                argmaxs = weights.multinomial(1)
                inter_nodes.append(
                    sum(
                        layer(nodes[j], nodes_accept_op_idx[j], first_layer) * w
                        for layer, w in zip(self.edges[node_str], weights)
                    )
                )
                new_nodes_accept_op_idx.append(argmaxs)
            nodes_accept_op_idx.append(new_nodes_accept_op_idx)
            nodes.append(sum(inter_nodes))
        return nodes[-1]

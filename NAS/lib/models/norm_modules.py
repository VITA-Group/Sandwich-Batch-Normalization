import torch.nn as nn
import torch


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, affine, track_running_stats, num_prev_ops: int, num_prev_nodes: int,
                 is_node_zero: bool):
        super().__init__()
        assert not affine
        num_classes = num_prev_ops ** num_prev_nodes
        self.num_prev_ops = num_prev_ops
        self.is_node_zero = is_node_zero
        self.num_features = num_features

        self.bn = nn.BatchNorm2d(num_features, affine=affine, track_running_stats=track_running_stats)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, input, prev_op_idxs=[0], first_layer=False):
        batch_size = input.size(0)
        cls_idx = sum([(self.num_prev_ops ** idx) * prev_op_idx for idx, prev_op_idx in enumerate(prev_op_idxs)])
        cls_idx_vector = torch.tensor([cls_idx]).repeat(batch_size, 1)
        if input.is_cuda:
            cls_idx_vector = cls_idx_vector.cuda()
        norm_output = self.bn(input)

        gamma, beta = self.embed(cls_idx_vector).chunk(2, -1)
        out = gamma.view(-1, self.num_features, 1, 1) * norm_output + beta.view(-1, self.num_features, 1, 1)
        return out


class SandwichBatchNorm2d(nn.Module):
    def __init__(self, num_features, affine, track_running_stats, num_prev_ops: int, num_prev_nodes: int,
                 is_node_zero: bool):
        super().__init__()
        assert affine
        num_classes = num_prev_ops ** num_prev_nodes
        self.num_prev_ops = num_prev_ops
        self.is_node_zero = is_node_zero
        self.num_features = num_features

        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features, affine=False, track_running_stats=track_running_stats) for _ in
             range(num_classes)])
        self.shared_weight = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.shared_bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)

        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, input, prev_op_idxs=[0], first_layer=False):
        batch_size = input.size(0)
        cls_idx = sum([(self.num_prev_ops ** idx) * prev_op_idx for idx, prev_op_idx in enumerate(prev_op_idxs)])
        cls_idx_vector = torch.tensor([cls_idx]).repeat(batch_size, 1)
        if input.is_cuda:
            cls_idx_vector = cls_idx_vector.cuda()
        norm_output = self.bns[cls_idx](input)
        shared_output = norm_output * self.shared_weight.view(1, self.num_features, 1, 1).expand(input.size()) + \
                      self.shared_bias.view(1, self.num_features, 1, 1).expand(input.size())
        if first_layer and self.is_node_zero:
            return shared_output
        else:
            gamma, beta = self.embed(cls_idx_vector).chunk(2, -1)
            out = gamma.view(-1, self.num_features, 1, 1) * shared_output + beta.view(-1, self.num_features, 1, 1)
            return out

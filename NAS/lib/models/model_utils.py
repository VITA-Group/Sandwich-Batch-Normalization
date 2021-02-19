# -*- coding: utf-8 -*-
# @Date    : 4/11/20
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0
import torch
import torch.nn.functional as F


def cal_entropy(logit: torch.Tensor, dim=-1) -> torch.Tensor:
    """
    ~
    :param logit: An unnormalized vector.
    :param dim: ~
    :return: entropy
    """
    prob = F.softmax(logit, dim=dim)
    log_prob = F.log_softmax(logit, dim=dim)

    entropy = -(log_prob * prob).sum(-1, keepdim=False)

    return entropy

# This file is for experimental usage
import torch

import torch.nn as nn
import torch.nn.functional as F


def random_alpha(model, epsilon):
    if isinstance(model, torch.nn.DataParallel):
        for p in model.module.get_alphas():
            p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
        model.module.clip()
    else:
        for p in model.get_alphas():
            p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
        model.clip()


def sanitize_bn(model):
    if isinstance(model, nn.DataParallel):
        for m in model.module.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean = torch.zeros_like(m.running_mean)
                m.running_var = torch.ones_like(m.running_var)
    else:
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean = torch.zeros_like(m.running_mean)
            m.running_var = torch.ones_like(m.running_var)


def recal_bn(network, loader):
    print("=> recal BN")
    sanitize_bn(network)
    network.train()
    with torch.no_grad():
        for step, (arch_inputs, _) in enumerate(loader):
            arch_inputs = arch_inputs.cuda(non_blocking=True)
            _, _ = network(arch_inputs)
            del arch_inputs


def get_per_egde_value_dict(arch_param, normalize=True):
    arch_param_dict = {}
    for edge_idx, edge_param in enumerate(arch_param):
        if normalize:
            edge_param = F.softmax(edge_param, dim=-1)
        edge_param_list = edge_param.cpu().detach().numpy().tolist()
        edge_dict = {f"op_{op_idx}": val for op_idx, val in enumerate(edge_param_list)}
        arch_param_dict[f"edge_{edge_idx}"] = edge_dict
    return arch_param_dict

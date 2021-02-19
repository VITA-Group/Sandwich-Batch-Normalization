# -*- coding: utf-8 -*-
# @Date    : 2/16/21
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import argparse
import logging
import os
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from functions import validate, validate_adv
from models.network import get_network_func


def main(args):
    logging.info(args)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    cudnn.benchmark = True

    model = get_network_func(args.norm_module)
    model.cuda()

    print('load checkpoint')
    assert os.path.exists(args.weight_path), print(f"Cannot find {args.weight_path}")
    checkpoint = torch.load(args.weight_path)
    model.load_state_dict(checkpoint['state_dict'])

    # dataset process
    test_dataset = datasets.CIFAR10(root=args.data, train=False, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # evaluate on test set
    # clean branch
    print(f"=>test on clean branch")
    test_tacc_clean, test_tloss_clean = validate(args, test_loader, model, criterion, 0)
    print(f"Test accuracy: {test_tacc_clean}, Test loss: {test_tloss_clean}")
    # test_atacc_clean, test_atloss_clean = validate_adv(args, test_loader, model, criterion, 0)
    # print(f"Adversarial test accuracy: {test_atacc_clean}, Adversarial test loss: {test_atloss_clean}")

    # adv branch
    print(f"=>test on adv branch")
    test_tacc_adv, test_tloss_adv = validate(args, test_loader, model, criterion, 1)
    print(f"Test accuracy: {test_tacc_adv}, Test loss: {test_tloss_adv}")
    test_atacc_adv, test_atloss_adv = validate_adv(args, test_loader, model, criterion, 1)
    print(f"Adversarial test accuracy: {test_atacc_adv}, Adversarial test loss: {test_atloss_adv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')

    parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--norm_module', type=str, required=True, choices=["bn", "auxbn", "saauxbn"], help='model type')
    parser.add_argument('--print_freq', default=50, type=int, help='print frequency')

    # test PGD args
    parser.add_argument('--weight_path', required=True, type=str, help='the path of weight')
    parser.add_argument('--test_eps', default=8, type=float, help='test_eps')
    parser.add_argument('--test_gamma', default=2, type=float, help='test_gamma')
    parser.add_argument('--test_steps', default=10, type=int, help='test_steps')
    parser.add_argument('--test_randinit', action="store_true", help="test_randinit")
    args = parser.parse_args()
    main(args)

# -*- coding: utf-8 -*-
# @Date    : 2/16/21
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import argparse
import logging
import os
import pickle
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from functions import train, validate, validate_adv, save_checkpoint
from models.network import get_network_func


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    logging.info(args)
    writer = SummaryWriter(args.save_dir)

    best_prec1, best_ata, cln_best_prec1 = 0, 0, 0
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    cudnn.benchmark = True

    setup_seed(args.seed)

    model = get_network_func(args.norm_module)
    model.cuda()

    start_epoch = 0
    if args.resume:
        print('resume from checkpoint')
        checkpoint = torch.load(os.path.join(args.save_dir, 'model.pt'))
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    # dataset process
    train_datasets = datasets.CIFAR10(root=args.data, train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor()
    ]), download=True)

    test_dataset = datasets.CIFAR10(root=args.data, train=False, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    valid_size = 0.1
    indices = list(range(len(train_datasets)))
    split = int(np.floor(valid_size * len(train_datasets)))

    np.random.seed(args.seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        batch_size=args.batch_size, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        train_datasets,
        batch_size=args.batch_size, sampler=valid_sampler)

    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    print('starting adv training')

    all_result = {}
    train_acc = []
    ta0 = []
    ata0 = []
    ta1 = []
    ata1 = []

    test_ta0 = []
    test_ata0 = []
    test_ta1 = []
    test_ata1 = []

    if os.path.exists(args.save_dir) is not True:
        os.mkdir(args.save_dir)

    for epoch in tqdm(range(args.epochs)):

        if epoch < start_epoch:
            scheduler.step()
            continue

        print(optimizer.state_dict()['param_groups'][0]['lr'])
        acc, train_loss = train(args, train_loader, model, criterion, optimizer, epoch)
        writer.add_scalar('acc/train_acc', acc, epoch)
        writer.add_scalar('loss/train_loss', train_loss, epoch)

        # evaluate on validation set
        tacc_clean, tloss_clean = validate(args, val_loader, model, criterion, 0)
        writer.add_scalar('acc/val_tacc_clean', tacc_clean, epoch)
        writer.add_scalar('loss/val_tloss_clean', tloss_clean, epoch)

        atacc_clean, atloss_clean = validate_adv(args, val_loader, model, criterion, 0)
        writer.add_scalar('acc/val_atacc_clean', atacc_clean, epoch)
        writer.add_scalar('loss/val_atloss_clean', atloss_clean, epoch)

        tacc_adv, tloss_adv = validate(args, val_loader, model, criterion, 1)
        writer.add_scalar('acc/val_tacc_adv', tacc_adv, epoch)
        writer.add_scalar('loss/val_tloss_adv', tloss_adv, epoch)

        atacc_adv, atloss_adv = validate_adv(args, val_loader, model, criterion, 1)
        writer.add_scalar('acc/val_atacc_adv', atacc_adv, epoch)
        writer.add_scalar('loss/val_atloss_adv', atloss_adv, epoch)

        # evaluate on test set
        # clean branch
        test_tacc_clean, test_tloss_clean = validate(args, test_loader, model, criterion, 0)
        writer.add_scalar('acc/test_tacc_clean', test_tacc_clean, epoch)
        writer.add_scalar('loss/test_tloss_clean', test_tloss_clean, epoch)

        test_atacc_clean, test_atloss_clean = validate_adv(args, test_loader, model, criterion, 0)
        writer.add_scalar('acc/test_atacc_clean', test_atacc_clean, epoch)
        writer.add_scalar('loss/test_atloss_clean', test_atloss_clean, epoch)

        # adv branch
        test_tacc_adv, test_tloss_adv = validate(args, test_loader, model, criterion, 1)
        writer.add_scalar('acc/test_tacc_adv', test_tacc_adv, epoch)
        writer.add_scalar('loss/test_tloss_adv', test_tloss_adv, epoch)

        test_atacc_adv, test_atloss_adv = validate_adv(args, test_loader, model, criterion, 1)
        writer.add_scalar('acc/test_atacc_adv', test_atacc_adv, epoch)
        writer.add_scalar('loss/test_atloss_adv', test_atloss_adv, epoch)

        scheduler.step()

        train_acc.append(acc)
        ta0.append(tacc_clean)
        ata0.append(atacc_clean)
        ta1.append(tacc_adv)
        ata1.append(atacc_adv)

        test_ta0.append(test_tacc_clean)
        test_ata0.append(test_atacc_clean)
        test_ta1.append(test_tacc_adv)
        test_ata1.append(test_atacc_adv)

        # remember best prec@1 and save checkpoint
        is_best = tacc_adv > best_prec1
        best_prec1 = max(tacc_adv, best_prec1)

        ata_is_best = atacc_adv > best_ata
        best_ata = max(atacc_adv, best_ata)

        cln_is_best = tacc_clean > cln_best_prec1
        cln_best_prec1 = max(tacc_clean, cln_best_prec1)

        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'ata_best_prec1': best_ata,
            }, is_best, filename=os.path.join(args.save_dir, 'best_model.pt'))

        if cln_is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'ata_best_prec1': best_ata,
            }, is_best, filename=os.path.join(args.save_dir, 'clean_best_model.pt'))

        if ata_is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'ata_best_prec1': best_ata,
            }, is_best, filename=os.path.join(args.save_dir, 'ata_best_model.pt'))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'ata_best_prec1': best_ata,
        }, is_best, filename=os.path.join(args.save_dir, 'model.pt'))

        if epoch and epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'ata_best_prec1': best_ata,
            }, is_best, filename=os.path.join(args.save_dir, f'model_epoch{epoch}.pt'))

        all_result['train'] = train_acc
        all_result['test_ta0'] = test_ta0
        all_result['test_ata0'] = test_ata0
        all_result['test_ta1'] = test_ta1
        all_result['test_ata1'] = test_ata1
        all_result['ta0'] = ta0
        all_result['ata0'] = ata0
        all_result['ta1'] = ta1
        all_result['ata1'] = ata1

        pickle.dump(all_result, open(os.path.join(args.save_dir, 'result_c.pkl'), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')

    parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
    parser.add_argument('--decreasing_lr', default='50,150', help='decreasing strategy')
    parser.add_argument('--seed', default=10, type=int, help='random seed')

    parser.add_argument('--save_dir', help='The directory used to save the trained models', default='adv', type=str)
    parser.add_argument('--norm_module', type=str, required=True, choices=["bn", "auxbn", "saauxbn"], help='model type')
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")

    # training PGD args
    parser.add_argument('--train_eps', default=8, type=float, help='train_eps')
    parser.add_argument('--train_gamma', default=2, type=float, help='train_gamma')
    parser.add_argument('--train_steps', default=7, type=int, help='train_steps')
    parser.add_argument('--train_randinit', action="store_true", help="train_randinit")

    # test PGD args
    parser.add_argument('--test_eps', default=8, type=float, help='test_eps')
    parser.add_argument('--test_gamma', default=2, type=float, help='test_gamma')
    parser.add_argument('--test_steps', default=10, type=int, help='test_steps')
    parser.add_argument('--test_randinit', action="store_true", help="test_randinit")
    args = parser.parse_args()
    main(args)

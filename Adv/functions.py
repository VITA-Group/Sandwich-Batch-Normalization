# -*- coding: utf-8 -*-
# @Date    : 2/16/21
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import torch

from attack_algo import PGD


def train(args, train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):

        if epoch == 0:
            warmup_lr(i, optimizer)

        input = input.cuda()
        target = target.cuda()

        # adv samples
        input_adv = PGD(input, criterion,
                        y=target,
                        eps=(args.train_eps / 255),
                        model=model,
                        steps=args.train_steps,
                        gamma=(args.train_gamma / 255),
                        randinit=args.train_randinit,
                        flag=1)

        input_adv = input_adv.cuda()

        # compute output
        inputsall_clean = {'x': input, 'flag': 0}
        inputsall_adv = {'x': input_adv, 'flag': 1}
        output_clean = model(**inputsall_clean)
        output_adv = model(**inputsall_adv)

        loss = (criterion(output_clean, target) + criterion(output_adv, target)) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_adv.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, i, len(train_loader), loss=losses, top1=top1))

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg


@torch.no_grad()
def validate(args, val_loader, model, criterion, flag):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        # compute output
        inputsall = {'x': input, 'flag': flag}
        output = model(**inputsall)
        loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg, losses.avg


def validate_adv(args, val_loader, model, criterion, flag):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        # adv samples
        input_adv = PGD(input, criterion,
                        y=target,
                        eps=(args.test_eps / 255),
                        model=model,
                        steps=args.test_steps,
                        gamma=(args.test_gamma / 255),
                        randinit=args.test_randinit,
                        flag=flag)

        input_adv = input_adv.cuda()
        # compute output
        with torch.no_grad():
            inputsall = {'x': input_adv, 'flag': flag}
            output = model(**inputsall)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), loss=losses, top1=top1))

    print('ATA {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def warmup_lr(step, optimizer):
    lr = 0.01 + step * (0.1 - 0.01) / 200
    lr = min(lr, 0.1)
    for p in optimizer.param_groups:
        p['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

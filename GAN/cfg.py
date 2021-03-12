# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO'],
        help='log level')
    parser.add_argument(
        '--max_epoch',
        type=int,
        default=200,
        help='number of epochs of training')
    parser.add_argument(
        '--max_iter',
        type=int,
        default=None,
        help='set the max iteration number')
    parser.add_argument(
        '-gen_bs',
        '--gen_batch_size',
        type=int,
        default=64,
        help='size of the batches')
    parser.add_argument(
        '-dis_bs',
        '--dis_batch_size',
        type=int,
        default=64,
        help='size of the batches')
    parser.add_argument(
        '--g_lr',
        type=float,
        default=0.0002,
        help='adam: gen learning rate')
    parser.add_argument(
        '--d_lr',
        type=float,
        default=0.0002,
        help='adam: disc learning rate')
    parser.add_argument(
        '--arch_g_lr',
        type=float,
        default=2e-4,
        help='adam: gen learning rate')
    parser.add_argument(
        '--arch_d_lr',
        type=float,
        default=2e-4,
        help='adam: disc learning rate')
    parser.add_argument(
        '--lr_decay',
        action='store_true',
        help='learning rate decay or not')
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.0,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.9,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--arch_beta1',
        type=float,
        default=0.5,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--arch_beta2',
        type=float,
        default=0.999,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=2,
        help='number of cpu threads to use during batch generation')
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=128,
        help='dimensionality of the latent space')
    parser.add_argument(
        '--channels',
        type=int,
        default=3,
        help='number of image channels')
    parser.add_argument(
        '--n_critic',
        type=int,
        default=1,
        help='number of training steps for discriminator per iter')
    parser.add_argument(
        '--val_freq',
        type=int,
        default=20,
        help='interval between each validation')
    parser.add_argument(
        '--print_freq',
        type=int,
        default=50,
        help='interval between each verbose')
    parser.add_argument(
        '--load_path',
        type=str,
        help='The reload model path')
    parser.add_argument(
        '--exp_name',
        type=str,
        help='The name of exp')
    parser.add_argument(
        '--d_sn',
        type=str2bool,
        default=False,
        help='add spectral_norm on discriminator?')
    parser.add_argument(
        '--g_sn',
        type=str2bool,
        default=False,
        help='add spectral_norm on generator?')
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=['cifar10', 'imagenet'],
        help='dataset type')
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data',
        help='The path of data set')
    parser.add_argument(
        '--path_file',
        type=str,
        help='The path of data set')
    parser.add_argument('--init_type', type=str, default='normal',
                        choices=['normal', 'orth', 'xavier_uniform', 'false'],
                        help='The init type')
    parser.add_argument('--gf_dim', type=int, default=64,
                        help='The base channel num of gen')
    parser.add_argument('--df_dim', type=int, default=64,
                        help='The base channel num of disc')
    parser.add_argument('--stop_search_epoch', type=int, default=-1,
                        help='The epoch of stopping search')
    parser.add_argument(
        '--model',
        type=str,
        default='sngan',
        choices=["sngan", "autogan"],
        help='model name')
    parser.add_argument(
        '--norm_module',
        type=str,
        required=True,
        choices=["ccbn", "sabn"],
        help='norm type')
    parser.add_argument('--eval_batch_size', type=int, default=100)
    parser.add_argument('--num_eval_imgs', type=int, default=50000)
    parser.add_argument(
        '--bottom_width',
        type=int,
        default=4,
        help="the base resolution of the GAN")
    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=128,
        help='size of the batches')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clip value')
    parser.add_argument(
        '--loss',
        type=str,
        default='hinge',
        choices=['hinge'],
        help='loss type')
    # parser.add_argument('--n_classes', type=int, default=0)
    parser.add_argument(
        '--snapshot',
        type=int,
        default=0,
        help='snapshot or not')
    opt = parser.parse_args()

    return opt

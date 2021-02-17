import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset import FlatFolderDataset, InfiniteSamplerWrapper, train_transform
from models import SaAdaINNet, AdaINNet, vgg, decoder
from functions import train_func, val_func

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', type=str, required=True,
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str, required=True,
                        help='Directory path to a batch of style images')
    parser.add_argument('--val_content_dir', type=str, required=True,
                        help='Directory path to a batch of content images')
    parser.add_argument('--val_style_dir', type=str, required=True,
                        help='Directory path to a batch of style images')
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

    # training options
    parser.add_argument('--save_dir', required=True,
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=10000)
    parser.add_argument('--val_interval', type=int, default=10000)
    parser.add_argument('--rand_seed', type=int, default=777, help='manual seed')
    parser.add_argument(
        '--saadain',
        action='store_true',
        help='using sandwich adain or not')

    args = parser.parse_args()
    prepare_seed(args.rand_seed)

    device = torch.device('cuda')
    save_dir = Path(args.save_dir + '_seed' + str(args.rand_seed))
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f'=> save_dir: {str(save_dir)}')
    writer = SummaryWriter(log_dir=str(save_dir))

    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    if args.saadain:
        network = SaAdaINNet(vgg, decoder)
    else:
        network = AdaINNet(vgg, decoder)
    network.train()
    network.to(device)

    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

    val_content_dataset = FlatFolderDataset(args.val_content_dir, content_tf)
    val_style_dataset = FlatFolderDataset(args.val_style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))

    if args.saadain:
        params = list(network.decoder.parameters()) + list(network.style_norm.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr)
    else:
        optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

    for num_iter in tqdm(range(args.max_iter)):
        train_func(args, network, optimizer, style_iter, content_iter, device, num_iter, save_dir, writer)
        if (num_iter + 1) % args.val_interval == 0 or (num_iter + 1) == args.val_interval or num_iter == 0:
            val_content_iter = iter(data.DataLoader(
                val_content_dataset, batch_size=args.batch_size,
                num_workers=args.n_threads))
            val_style_iter = iter(data.DataLoader(
                val_style_dataset, batch_size=args.batch_size,
                num_workers=args.n_threads))
            val_func(args, network, val_style_iter, val_content_iter, device, num_iter, writer)
            del val_content_iter
            del val_style_iter

    if writer:
        writer.close()

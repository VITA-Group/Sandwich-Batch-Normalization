# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

classes_dog_and_cat = np.arange(151, 294, dtype=np.int32)
class_to_index = dict()
for i, c in enumerate(classes_dog_and_cat):
    class_to_index[c] = i


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ShortSideCrop(object):
    def __call__(self, img):
        image_width, image_height = img.size
        short_side = image_height if image_height < image_width else image_width
        crop_size = short_side
        # Crop the center
        top = (image_height - crop_size) // 2
        left = (image_width - crop_size) // 2
        return F.crop(img, top, left, crop_size, crop_size)


class ImagenetDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, path_file, loader=default_loader, transform=None):
        self.transform = transform
        self.loader = loader
        file = open(path_file, "r")
        lines = file.readlines()
        self.img_path = []
        self.cls = []
        for line in lines:
            path, cls = line[: len(line) - 1].split(" ")
            self.img_path.append(os.path.join(data_root, path))
            self.cls.append(int(cls))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        cls_idx = class_to_index[self.cls[idx]]
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, cls_idx


class ImageDataset(object):
    def __init__(self, args):
        if args.dataset.lower() == "cifar10":
            args.n_classes = 10
            img_size = 32
            Dt = torchvision.datasets.CIFAR10
            transform = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

            # dataset
            train_dataset = Dt(
                root=args.data_path, train=True, transform=transform, download=True
            )
            valid_dataset = Dt(root=args.data_path, train=False, transform=transform)
            self.train = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.dis_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            self.test = self.valid

        elif args.dataset.lower() == "imagenet":
            args.n_classes = 143
            img_size = 128
            Dt = ImagenetDataset
            transform = transforms.Compose(
                [
                    ShortSideCrop(),
                    transforms.Resize(size=(img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            # dataset
            train_dataset = Dt(args.data_path, args.path_file, transform=transform)
            self.train = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.dis_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        else:
            raise NotImplementedError(f"Unknown dataset: {args.dataset}.")

#!/usr/bin/env bash

python train.py --content_dir ~/dataset/coco/images/train2017 --style_dir ~/dataset/paint/train_1 \
    --val_content_dir ~/dataset/coco/images/val2017 --val_style_dir ~/dataset/paint/test \
    --vgg pretrained_weights/vgg_normalised.pth --save_dir experiments/adain
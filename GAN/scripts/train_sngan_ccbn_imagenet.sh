#!/usr/bin/env bash

python train.py \
-gen_bs 32 \
-dis_bs 16 \
--dataset imagenet \
--data_path data/imagenet_sngan \
--path_file image_list_dog_and_cat.txt \
--max_iter 450000 \
--num_eval_imgs 20000 \
--model sngan \
--norm_module ccbn \
--latent_dim 128 \
--gf_dim 64 \
--df_dim 64 \
--g_sn False \
--d_sn True \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--n_critic 5 \
--val_freq 10 \
--exp_name sngan_ccbn
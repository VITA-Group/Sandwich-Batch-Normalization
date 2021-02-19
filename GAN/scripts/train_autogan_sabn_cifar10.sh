#!/usr/bin/env bash

python train.py \
-gen_bs 128 \
-dis_bs 64 \
--dataset cifar10 \
--max_iter 50000 \
--model autogan \
--norm_module sabn \
--latent_dim 128 \
--gf_dim 256 \
--df_dim 128 \
--g_sn False \
--d_sn True \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--n_critic 5 \
--val_freq 20 \
--exp_name autogan_sabn
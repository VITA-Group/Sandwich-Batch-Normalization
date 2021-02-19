#!/usr/bin/env bash

python test.py \
-gen_bs 128 \
-dis_bs 64 \
--dataset cifar10 \
--model sngan \
--norm_module sabn \
--latent_dim 128 \
--gf_dim 256 \
--df_dim 128 \
--g_sn False \
--d_sn True \
--load_path zoo/sngan_sabn_cifar10_gen_state_dict.pth \
--exp_name test_sngan_sabn
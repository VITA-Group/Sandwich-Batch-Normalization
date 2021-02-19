#!/usr/bin/env bash

python test.py \
-gen_bs 32 \
-dis_bs 16 \
--dataset imagenet \
--num_eval_imgs 20000 \
--model sngan \
--norm_module sabn \
--latent_dim 128 \
--gf_dim 64 \
--df_dim 64 \
--g_sn False \
--d_sn True \
--load_path zoo/sngan_sabn_imagenet_gen_state_dict.pth \
--exp_name test_sngan_sabn
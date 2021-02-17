python train.py --data ~/dataset --batch_size 128\
	--lr 0.1 --epochs 200 --seed 10 \
	--train_eps 8 --train_gamma 2 --train_steps 7 --train_randinit \
	--test_eps 8 --test_gamma 2 --test_steps 10 --test_randinit \
	--norm_module bn --save_dir output/bn
#!/bin/bash
# bash scripts/DARTS-V1-sabn.sh cifar100 0 777
# bash scripts/DARTS-V1-sabn.sh ImageNet16-120 0 777
echo script name: $0
echo $# arguments
if [ "$#" -lt 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need at least 4 parameters for dataset, tracking_status and seed"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

dataset=$1
BN=$2
seed=$3
early_stop_epoch=${4:-35}
w_lr=${5:-0.025}
space=${6:-nas-bench-201}

channel=16
num_cells=5
max_nodes=4

if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
  data_path="$TORCH_HOME/cifar.python"
else
  data_path="$TORCH_HOME/cifar.python/ImageNet16"
fi

benchmark_file=${TORCH_HOME}/NAS-Bench-201-v1_0-e61699.pth
#benchmark_file=${TORCH_HOME}/NAS-Bench-201-v1_1-096897.pth

save_dir=./output/search-cell-${space}/DARTS-V1/${dataset}/${seed}/DARTS-sabn-BN${BN}-es_epoch${early_stop_epoch}

OMP_NUM_THREADS=4 python DARTS-V1.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name ${space} \
	--config_path configs/nas-benchmark/algos/DARTS.config \
	--arch_nas_dataset ${benchmark_file} \
	--track_running_stats ${BN} --affine 1 --model DARTS-V1-sabn \
	--arch_learning_rate 0.0003 --arch_weight_decay 0.001 --weight_learning_rate ${w_lr} \
	--workers 4 --print_freq 200 --early_stop_epoch ${early_stop_epoch} --rand_seed ${seed}

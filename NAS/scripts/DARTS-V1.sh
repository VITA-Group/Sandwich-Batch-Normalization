#!/bin/bash
# bash scripts/DARTS-V1.sh cifar100 0 777 35
# bash scripts/DARTS-V1.sh ImageNet16-120 0 777 35
echo script name: $0
echo $# arguments
if [ "$#" -lt 4 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need at least 4 parameters for dataset, tracking_status, seed and epoch"
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
num_epochs=$4
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

if [ $num_epochs -ne 0 ]; then
  epoch_name="${num_epochs}Epoch"
else
  epoch_name="useConfigEpoch"
fi

benchmark_file=${TORCH_HOME}/NAS-Bench-201-v1_0-e61699.pth
#benchmark_file=${TORCH_HOME}/NAS-Bench-201-v1_1-096897.pth

save_dir=./output/search-cell-${space}/DARTS-V1/${dataset}/${seed}/DARTS-BN${BN}-${epoch_name}

OMP_NUM_THREADS=4 python DARTS-V1.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name ${space} \
	--config_path configs/nas-benchmark/algos/DARTS.config \
	--arch_nas_dataset ${benchmark_file} \
	--track_running_stats ${BN} \
	--arch_learning_rate 0.0003 --arch_weight_decay 0.001 --model DARTS-V1 \
	--workers 4 --print_freq 200 --num_epochs ${num_epochs} --weight_learning_rate ${w_lr} --rand_seed ${seed}

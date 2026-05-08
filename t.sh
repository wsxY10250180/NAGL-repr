#!/bin/bash
#SBATCH --job-name=wsx
#SBATCH --partition=mixed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --output=./outputs/train_on_visa.txt

python train.py \
  --data_root /gpool/home/wangshaoxiong/dataset \
  --fold 1 \
  --epoch 20 \
  --batch_size 8 \
  --image_size 448 \
  --print_freq 50 \
  --n_shot 1 \
  --a_shot 1 \
  --num_learnable_proxies 25 \
  --save_path ./outputs/train_on_visa

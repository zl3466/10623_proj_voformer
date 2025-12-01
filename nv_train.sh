#!/bin/bash
source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/etc/profile.d/conda.sh
conda activate voformer
export WANDB_API_KEY="de2b136779280d18cb6d59d1a23248b5010833f8"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    train.py \
    --nusc_data_path /lustre/fsw/portfolios/nvr/users/ymingli/datasets/vlm_visual_odom/NuScenes/train_test \
    --batch_size 8 \
    --vocab_size 100 \
    --wandb_project visual-odometry \
    --wandb_run_name nv_vocab100_img196
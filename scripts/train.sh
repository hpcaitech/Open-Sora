#!/usr/bin/env bash

# get root dir
FOLDER_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR=$FOLDER_DIR/..

# go to root dir
cd $ROOT_DIR

# define dataset shards
COLLATED_VIDEO_DIR=./dataset/MSRVTT-collated/val/videos
PROCESSED_DATASET=(
    ./dataset/MSRVTT-processed/val/part-00000
    ./dataset/MSRVTT-processed/val/part-00001
    ./dataset/MSRVTT-processed/val/part-00002
    ./dataset/MSRVTT-processed/val/part-00003
    ./dataset/MSRVTT-processed/val/part-00004
    ./dataset/MSRVTT-processed/val/part-00005
    ./dataset/MSRVTT-processed/val/part-00006
    ./dataset/MSRVTT-processed/val/part-00007
    ./dataset/MSRVTT-processed/val/part-00008
    ./dataset/MSRVTT-processed/val/part-00009
)

# run single node training
torchrun --standalone \
    --nproc_per_node 4 \
    train.py \
    --epochs 1 \
    --batch_size 1 \
    --lr 1e-4 \
    --accumulation_steps 32 \
    --grad_checkpoint \
    --dataset $PROCESSED_DATASET \
    --video_dir $COLLATED_VIDEO_DIR \
    --save_interval 224 \
    --checkpoint_dir ./checkpoints \
    --tensorboard_dir ./runs

#!/bin/bash

set -x
set -e

CMD="torchrun --standalone --nproc_per_node 1 eval/loss/eval_loss.py configs/opensora-v1-2/misc/eval_loss.py"
CKPT_PATH=$1
IMG_PATH="/mnt/nfs-207/sora_data/meta/img_1k.csv"
VID_PATH="/mnt/nfs-207/sora_data/meta/vid_100.csv"

if [[ $CKPT_PATH == *"ema"* ]]; then
    parentdir=$(dirname $CKPT_PATH)
    CKPT_BASE=$(basename $parentdir)_ema
else
    CKPT_BASE=$(basename $CKPT_PATH)
fi
LOG_BASE=logs/loss/${CKPT_BASE}
mkdir -p logs/loss
echo "Logging to $LOG_BASE"

CUDA_VISIBLE_DEVICES=0 $CMD --data-path $IMG_PATH --ckpt-path $CKPT_PATH >${LOG_BASE}_img.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $CMD --data-path $VID_PATH --ckpt-path $CKPT_PATH --resolution 144p >${LOG_BASE}_144p_vid.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $CMD --data-path $VID_PATH --ckpt-path $CKPT_PATH --resolution 240p >${LOG_BASE}_240p_vid.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 $CMD --data-path $VID_PATH --ckpt-path $CKPT_PATH --resolution 360p >${LOG_BASE}_360p_vid.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 $CMD --data-path $VID_PATH --ckpt-path $CKPT_PATH --resolution 480p >${LOG_BASE}_480p_vid.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 $CMD --data-path $VID_PATH --ckpt-path $CKPT_PATH --resolution 720p >${LOG_BASE}_720p_vid.log 2>&1 &

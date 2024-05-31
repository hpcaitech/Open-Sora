#!/bin/bash

set -x
set -e

CMD="torchrun --standalone --nproc_per_node 1 eval/loss/eval_loss.py configs/opensora-v1-2/misc/eval_loss.py"
CKPT_PATH=$1
MODEL_NAME=$2
IMG_PATH="/mnt/jfs-hdd/sora/meta/eval_loss/img_1k.csv"
VID_PATH="/mnt/jfs-hdd/sora/meta/eval_loss/vid_100.csv"

if [[ $CKPT_PATH == *"ema"* ]]; then
    parentdir=$(dirname $CKPT_PATH)
    CKPT_BASE=$(basename $parentdir)_ema
else
    CKPT_BASE=$(basename $CKPT_PATH)
fi
LOG_BASE=logs/loss/${MODEL_NAME}_${CKPT_BASE}
mkdir -p logs/loss
echo "Logging to $LOG_BASE"


GPUS=(1 2 3 4 5)
RESOLUTION=(144p 240p 360p 480p 720p)

CUDA_VISIBLE_DEVICES=0 $CMD --data-path $IMG_PATH --ckpt-path $CKPT_PATH >${LOG_BASE}_img.log 2>&1 &

for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[i]} $CMD --data-path $VID_PATH --ckpt-path $CKPT_PATH --resolution ${RESOLUTION[i]} >${LOG_BASE}_${RESOLUTION[i]}_vid.log 2>&1 &
done

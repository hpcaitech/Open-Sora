#!/bin/bash

CMD="torchrun --standalone --nproc_per_node 1 eval/loss/eval_loss.py configs/opensora-v1-2/misc/eval_loss.py"
CKPT_PATH=$1
MODEL_NAME=$2
IMG_PATH=$3
VID_PATH=$4

if [ -z $IMG_PATH ]; then
    IMG_PATH="/mnt/jfs-hdd/sora/meta/validation/img_1k.csv"
fi

if [ -z $VID_PATH ]; then
    VID_PATH="/mnt/jfs-hdd/sora/meta/validation/vid_100.csv"
fi

if [[ $CKPT_PATH == *"ema"* ]]; then
    parentdir=$(dirname $CKPT_PATH)
    CKPT_BASE=$(basename $parentdir)_ema
else
    CKPT_BASE=$(basename $CKPT_PATH)
fi
LOG_BASE=$(dirname $CKPT_PATH)/eval
mkdir -p $LOG_BASE
echo "Logging to $LOG_BASE"


GPUS=(3 4 5 6 7)
RESOLUTION=(144p 240p 360p 480p 720p)

CUDA_VISIBLE_DEVICES=0 $CMD --data-path $IMG_PATH --ckpt-path $CKPT_PATH --start-index 0 --end-index 5 >${LOG_BASE}/img_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $CMD --data-path $IMG_PATH --ckpt-path $CKPT_PATH --start-index 5 --end-index 6 >${LOG_BASE}/img_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $CMD --data-path $IMG_PATH --ckpt-path $CKPT_PATH --start-index 6 >${LOG_BASE}/img_2.log 2>&1 &


for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[i]} $CMD --data-path $VID_PATH --ckpt-path $CKPT_PATH --resolution ${RESOLUTION[i]} >${LOG_BASE}/${RESOLUTION[i]}_vid.log 2>&1 &
done

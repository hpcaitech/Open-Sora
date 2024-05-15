#!/bin/bash

set -x
set -e

CKPT=$1
NUM_FRAMES=$2
MODEL_NAME=$3

if [[ $CKPT == *"ema"* ]]; then
    parentdir=$(dirname $CKPT)
    CKPT_BASE=$(basename $parentdir)_ema
else
    CKPT_BASE=$(basename $CKPT)
fi
LOG_BASE=logs/sample/${MODEL_NAME}_${CKPT_BASE}
echo "Logging to $LOG_BASE"

# == sample & human evaluation ==
CUDA_VISIBLE_DEVICES=0 bash eval/sample.sh $CKPT 1 $MODEL_NAME -1 >${LOG_BASE}_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash eval/sample.sh $CKPT $NUM_FRAMES $MODEL_NAME -2a >${LOG_BASE}_2a.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 bash eval/sample.sh $CKPT $NUM_FRAMES $MODEL_NAME -2b >${LOG_BASE}_2b.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 bash eval/sample.sh $CKPT $NUM_FRAMES $MODEL_NAME -2c >${LOG_BASE}_2c.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 bash eval/sample.sh $CKPT $NUM_FRAMES $MODEL_NAME -2d >${LOG_BASE}_2d.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 bash eval/sample.sh $CKPT $NUM_FRAMES $MODEL_NAME -2e >${LOG_BASE}_2e.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 bash eval/sample.sh $CKPT $NUM_FRAMES $MODEL_NAME -2f >${LOG_BASE}_2f.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 bash eval/sample.sh $CKPT $NUM_FRAMES $MODEL_NAME -2g >${LOG_BASE}_2g.log 2>&1 &

# CUDA_VISIBLE_DEVICES=7 bash eval/sample.sh $CKPT $NUM_FRAMES $MODEL_NAME -2h >${LOG_BASE}_2h.log 2>&1 &

# kill all by: pkill -f "inference"

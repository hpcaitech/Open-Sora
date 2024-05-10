#!/bin/bash

set -x
set -e

CKPT=$1
CUDA_NUM=$2
NUM_FRAMES=$3
MODEL_NAME=$4


if [[ $CKPT == *"ema"* ]]; then
    parentdir=$(dirname $CKPT)
    CKPT_BASE=$(basename $parentdir)_ema
else
    CKPT_BASE=$(basename $CKPT)
fi

LOG_BASE=logs/sample/${MODEL_NAME}_${CKPT_BASE}
echo "Logging to $LOG_BASE"

# == sample & human evaluation ==
echo "running image task"
CUDA_VISIBLE_DEVICES=$CUDA_NUM bash eval/sample.sh $CKPT 1 $MODEL_NAME -1 >${LOG_BASE}_1.log 2>&1
echo "running task 2a"
CUDA_VISIBLE_DEVICES=$CUDA_NUM bash eval/sample.sh $CKPT $NUM_FRAMES $MODEL_NAME -2a >${LOG_BASE}_2a.log 2>&1
# echo "running task 2b"
# CUDA_VISIBLE_DEVICES=$CUDA_NUM bash eval/sample.sh $CKPT $NUM_FRAMES $MODEL_NAME -2b >${LOG_BASE}_2b.log 2>&1
# echo "running task 2c"
# CUDA_VISIBLE_DEVICES=$CUDA_NUM bash eval/sample.sh $CKPT $NUM_FRAMES $MODEL_NAME -2c >${LOG_BASE}_2c.log 2>&1
# echo "running task 2d"
# CUDA_VISIBLE_DEVICES=$CUDA_NUM bash eval/sample.sh $CKPT $NUM_FRAMES $MODEL_NAME -2d >${LOG_BASE}_2d.log 2>&1
# echo "running task 2e"
# CUDA_VISIBLE_DEVICES=$CUDA_NUM bash eval/sample.sh $CKPT $NUM_FRAMES $MODEL_NAME -2e >${LOG_BASE}_2e.log 2>&1
# echo "running task 2f"
# CUDA_VISIBLE_DEVICES=$CUDA_NUM bash eval/sample.sh $CKPT $NUM_FRAMES $MODEL_NAME -2f >${LOG_BASE}_2f.log 2>&1
# echo "running task 2g"
# CUDA_VISIBLE_DEVICES=$CUDA_NUM bash eval/sample.sh $CKPT $NUM_FRAMES $MODEL_NAME -2g >${LOG_BASE}_2g.log 2>&1
# echo "running task 2h"
# CUDA_VISIBLE_DEVICES=$CUDA_NUM bash eval/sample.sh $CKPT $NUM_FRAMES $MODEL_NAME -2h >${LOG_BASE}_2h.log 2>&1

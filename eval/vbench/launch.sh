#!/bin/bash

set -x
set -e

CKPT=$1
MODEL_NAME=$2
NUM_FRAMES=$3

if [[ $CKPT == *"ema"* ]]; then
    parentdir=$(dirname $CKPT)
    CKPT_BASE=$(basename $parentdir)_ema
else
    CKPT_BASE=$(basename $CKPT)
fi
LOG_BASE=logs/sample/${MODEL_NAME}_${CKPT_BASE}
echo "Logging to $LOG_BASE"

GPUS=(0 1 2 3 4 5 6 7)
TASK_ID_LIST=(4a 4b 4c 4d 4e 4f 4g 4h)

for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[i]} bash eval/sample.sh $CKPT ${NUM_FRAMES} ${MODEL_NAME} -${TASK_ID_LIST[i]} >${LOG_BASE}_${TASK_ID_LIST[i]}.log 2>&1 &
done

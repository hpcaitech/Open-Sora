#!/bin/bash

CKPT=$1
NUM_FRAMES=$2
MODEL_NAME=$3

if [[ $CKPT == *"ema"* ]]; then
    parentdir=$(dirname $CKPT)
    CKPT_BASE=$(basename $parentdir)_ema
else
    CKPT_BASE=$(basename $CKPT)
fi
LOG_BASE=$(dirname $CKPT)/eval
mkdir -p ${LOG_BASE}
echo "Logging to $LOG_BASE"

GPUS=(0 1 2 3 4 5 6 7)
# TASK_ID_LIST=(1 2a 2b 2c 2d 2e 2f 2g) # move image to video task
TASK_ID_LIST=(2a 2b 2c 2d 2e 2f 2g 2h)
# FRAME_LIST=(1 $NUM_FRAMES $NUM_FRAMES $NUM_FRAMES $NUM_FRAMES $NUM_FRAMES $NUM_FRAMES $NUM_FRAMES)

for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[i]} bash eval/sample.sh $CKPT $NUM_FRAMES $MODEL_NAME -${TASK_ID_LIST[i]} >${LOG_BASE}/${TASK_ID_LIST[i]}.log 2>&1 &
done

# kill all by: pkill -f "inference"

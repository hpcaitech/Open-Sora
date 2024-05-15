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

CUDA_VISIBLE_DEVICES=0 bash eval/sample.sh $CKPT -4a >${LOG_BASE}_4a.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash eval/sample.sh $CKPT -4b >${LOG_BASE}_4b.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 bash eval/sample.sh $CKPT -4c >${LOG_BASE}_4c.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 bash eval/sample.sh $CKPT -4d >${LOG_BASE}_4d.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 bash eval/sample.sh $CKPT -4e >${LOG_BASE}_4e.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 bash eval/sample.sh $CKPT -4f >${LOG_BASE}_4f.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 bash eval/sample.sh $CKPT -4g >${LOG_BASE}_4g.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 bash eval/sample.sh $CKPT -4h >${LOG_BASE}_4h.log 2>&1 &

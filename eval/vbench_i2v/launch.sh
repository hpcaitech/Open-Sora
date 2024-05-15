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

CUDA_VISIBLE_DEVICES=0 bash eval/sample.sh $CKPT -5a >${LOG_BASE}_5a.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash eval/sample.sh $CKPT -5b >${LOG_BASE}_5b.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 bash eval/sample.sh $CKPT -5c >${LOG_BASE}_5c.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 bash eval/sample.sh $CKPT -5d >${LOG_BASE}_5d.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 bash eval/sample.sh $CKPT -5e >${LOG_BASE}_5e.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 bash eval/sample.sh $CKPT -5f >${LOG_BASE}_5f.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 bash eval/sample.sh $CKPT -5g >${LOG_BASE}_5g.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 bash eval/sample.sh $CKPT -5h >${LOG_BASE}_5h.log 2>&1 &

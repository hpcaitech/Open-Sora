#!/bin/bash

set -x
set -e

CKPT=$1
if [[ $CKPT == *"ema"* ]]; then
    parentdir=$(dirname $CKPT)
    CKPT_BASE=$(basename $parentdir)_ema
else
    CKPT_BASE=$(basename $CKPT)
fi
LOG_BASE=logs/sample/CKPT_BASE
echo "Logging to $LOG_BASE"

CUDA_VISIBLE_DEVICES=0 bash eval/sample.sh $CKPT -1 >${LOG_BASE}_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash eval/sample.sh $CKPT -3 >${LOG_BASE}_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 bash eval/sample.sh $CKPT -2a >${LOG_BASE}_2a.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 bash eval/sample.sh $CKPT -2b >${LOG_BASE}_2b.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 bash eval/sample.sh $CKPT -2c >${LOG_BASE}_2c.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 bash eval/sample.sh $CKPT -2d >${LOG_BASE}_2d.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 bash eval/sample.sh $CKPT -2e >${LOG_BASE}_2e.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 bash eval/sample.sh $CKPT -2f >${LOG_BASE}_2f.log 2>&1 &

# kill all by: pkill -f "inference"

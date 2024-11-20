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
LOG_BASE=logs/sample/$CKPT_BASE
echo "Logging to $LOG_BASE"

# == sample & human evaluation ==
# CUDA_VISIBLE_DEVICES=0 bash eval/sample.sh $CKPT -1 >${LOG_BASE}_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 bash eval/sample.sh $CKPT -2a >${LOG_BASE}_2a.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 bash eval/sample.sh $CKPT -2b >${LOG_BASE}_2b.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 bash eval/sample.sh $CKPT -2c >${LOG_BASE}_2c.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 bash eval/sample.sh $CKPT -2d >${LOG_BASE}_2d.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 bash eval/sample.sh $CKPT -2e >${LOG_BASE}_2e.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 bash eval/sample.sh $CKPT -2f >${LOG_BASE}_2f.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 bash eval/sample.sh $CKPT -2g >${LOG_BASE}_2g.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 bash eval/sample.sh $CKPT -2h >${LOG_BASE}_2h.log 2>&1 &

# == vbench ==
# CUDA_VISIBLE_DEVICES=0 bash eval/sample.sh $CKPT -4a >${LOG_BASE}_4a.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 bash eval/sample.sh $CKPT -4b >${LOG_BASE}_4b.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 bash eval/sample.sh $CKPT -4c >${LOG_BASE}_4c.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 bash eval/sample.sh $CKPT -4d >${LOG_BASE}_4d.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 bash eval/sample.sh $CKPT -4e >${LOG_BASE}_4e.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 bash eval/sample.sh $CKPT -4f >${LOG_BASE}_4f.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 bash eval/sample.sh $CKPT -4g >${LOG_BASE}_4g.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 bash eval/sample.sh $CKPT -4h >${LOG_BASE}_4h.log 2>&1 &

# == vbench i2v ==
# CUDA_VISIBLE_DEVICES=0 bash eval/sample.sh $CKPT -5a >${LOG_BASE}_5a.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 bash eval/sample.sh $CKPT -5b >${LOG_BASE}_5b.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 bash eval/sample.sh $CKPT -5c >${LOG_BASE}_5c.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 bash eval/sample.sh $CKPT -5d >${LOG_BASE}_5d.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 bash eval/sample.sh $CKPT -5e >${LOG_BASE}_5e.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 bash eval/sample.sh $CKPT -5f >${LOG_BASE}_5f.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 bash eval/sample.sh $CKPT -5g >${LOG_BASE}_5g.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 bash eval/sample.sh $CKPT -5h >${LOG_BASE}_5h.log 2>&1 &

# kill all by: pkill -f "inference"

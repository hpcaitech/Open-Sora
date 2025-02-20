#!/bin/bash

CMD_FILE_CONFIG="eval/loss/eval_loss.py configs/opensora-pro/misc/eval_loss.py"
PORTS=$1
CKPT_PATH=$2
VID_PATH=$3

# only evaluate for 360p, 102f
RESOLUTION=360p
torchrun --master-port ${PORTS} --nproc_per_node 1 $CMD_FILE_CONFIG --data-path $VID_PATH --ckpt-path $CKPT_PATH --resolution ${RESOLUTION}

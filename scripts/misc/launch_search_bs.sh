#!/bin/bash

set -x
set -e

CMD="torchrun --standalone --nproc_per_node 1 scripts/misc/search_bs.py configs/opensora-v1-2/misc/bs.py"
DATA_PATH="/mnt/nfs-207/sora_data/meta/searchbs.csv"

LOG_BASE=logs/search_bs
mkdir -p logs/search_bs
echo "Logging to $LOG_BASE"

CUDA_VISIBLE_DEVICES=0 $CMD --data-path $DATA_PATH --resolution 144p >${LOG_BASE}/144p.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $CMD --data-path $DATA_PATH --resolution 240p >${LOG_BASE}/240p.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $CMD --data-path $DATA_PATH --resolution 512 >${LOG_BASE}/512.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $CMD --data-path $DATA_PATH --resolution 480p >${LOG_BASE}/480p.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 $CMD --data-path $DATA_PATH --resolution 1024 >${LOG_BASE}/1024.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 $CMD --data-path $DATA_PATH --resolution 1080p >${LOG_BASE}/1080p.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 $CMD --data-path $DATA_PATH --resolution 2048 >${LOG_BASE}/2048.log 2>&1 &

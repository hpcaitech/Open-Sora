#!/bin/bash

set -x
set -e

START_SPLIT=0
NUM_SPLIT=100

DATA_PATH=$1
DATA_ARG="--data-path $DATA_PATH"

CMD="torchrun --standalone --nproc_per_node 1 scripts/misc/extract_feat.py configs/opensora-v1-2/misc/extract.py $DATA_ARG"
declare -a GPUS=(0 1 2 3 4 5 6 7)

mkdir -p logs/extract_feat

for i in "${GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=$i $CMD --start-index $(($START_SPLIT + i * $NUM_SPLIT)) --end-index $(($START_SPLIT + (i + 1) * $NUM_SPLIT)) >logs/extract_feat/$i.log 2>&1 &
done

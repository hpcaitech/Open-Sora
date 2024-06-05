#!/bin/bash

set -x
set -e

TEXT_PATH=/home/data/sora_data/pixart-sigma-generated/text.txt
OUTPUT_PATH=/home/data/sora_data/pixart-sigma-generated/raw
CMD="python scripts/inference.py configs/pixart/inference/1x2048MS.py"
# LOG_BASE=logs/sample/generate
LOG_BASE=$(dirname $CKPT)/eval/generate
mkdir -p ${LOG_BASE}
NUM_PER_GPU=10000
N_LAUNCH=2
NUM_START=$(($N_LAUNCH * $NUM_PER_GPU * 8))

CUDA_VISIBLE_DEVICES=0 $CMD --prompt-path $TEXT_PATH --save-dir $OUTPUT_PATH --start-index $(($NUM_START + $NUM_PER_GPU * 0)) --end-index $(($NUM_START + $NUM_PER_GPU * 1)) --image-size 2048 2048 --verbose 1 --batch-size 2 >${LOG_BASE}/${N_LAUNCH}_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $CMD --prompt-path $TEXT_PATH --save-dir $OUTPUT_PATH --start-index $(($NUM_START + $NUM_PER_GPU * 1)) --end-index $(($NUM_START + $NUM_PER_GPU * 2)) --image-size 1408 2816 --verbose 1 --batch-size 2 >${LOG_BASE}/${N_LAUNCH}_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $CMD --prompt-path $TEXT_PATH --save-dir $OUTPUT_PATH --start-index $(($NUM_START + $NUM_PER_GPU * 2)) --end-index $(($NUM_START + $NUM_PER_GPU * 3)) --image-size 2816 1408 --verbose 1 --batch-size 2 >${LOG_BASE}/${N_LAUNCH}_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $CMD --prompt-path $TEXT_PATH --save-dir $OUTPUT_PATH --start-index $(($NUM_START + $NUM_PER_GPU * 3)) --end-index $(($NUM_START + $NUM_PER_GPU * 4)) --image-size 1664 2304 --verbose 1 --batch-size 2 >${LOG_BASE}/${N_LAUNCH}_4.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 $CMD --prompt-path $TEXT_PATH --save-dir $OUTPUT_PATH --start-index $(($NUM_START + $NUM_PER_GPU * 4)) --end-index $(($NUM_START + $NUM_PER_GPU * 5)) --image-size 2304 1664 --verbose 1 --batch-size 2 >${LOG_BASE}/${N_LAUNCH}_5.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 $CMD --prompt-path $TEXT_PATH --save-dir $OUTPUT_PATH --start-index $(($NUM_START + $NUM_PER_GPU * 5)) --end-index $(($NUM_START + $NUM_PER_GPU * 6)) --image-size 1536 2560 --verbose 1 --batch-size 2 >${LOG_BASE}/${N_LAUNCH}_6.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 $CMD --prompt-path $TEXT_PATH --save-dir $OUTPUT_PATH --start-index $(($NUM_START + $NUM_PER_GPU * 6)) --end-index $(($NUM_START + $NUM_PER_GPU * 7)) --image-size 2560 1536 --verbose 1 --batch-size 2 >${LOG_BASE}/${N_LAUNCH}_7.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 $CMD --prompt-path $TEXT_PATH --save-dir $OUTPUT_PATH --start-index $(($NUM_START + $NUM_PER_GPU * 7)) --end-index $(($NUM_START + $NUM_PER_GPU * 8)) --image-size 2048 2048 --verbose 1 --batch-size 2 >${LOG_BASE}/${N_LAUNCH}_8.log 2>&1 &

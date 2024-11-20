#!/bin/bash

set -x
set -e

CKPT=$1
PROMPT=$2
NUM_SAMPLE=3
NAME=$(date +%Y%m%d%H%M%S)
CMD="python scripts/inference-long.py configs/opensora-v1-1/inference/sample.py"
if [[ $CKPT == *"ema"* ]]; then
    parentdir=$(dirname $CKPT)
    CKPT_BASE=$(basename $parentdir)_ema
else
    CKPT_BASE=$(basename $CKPT)
fi
OUTPUT="./samples/samples_${CKPT_BASE}_${NAME}"
start=$(date +%s)

# Generate samples

# == 16x240p ==
# 1:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x240p_1_1 \
    --num-frames 16 --image-size 320 320 --num-sample $NUM_SAMPLE
# 16:9
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x240p_16_9 \
    --num-frames 16 --image-size 240 426 --num-sample $NUM_SAMPLE
# 9:16
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x240p_9_16 \
    --num-frames 16 --image-size 426 240 --num-sample $NUM_SAMPLE
# 4:3
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x240p_4_3 \
    --num-frames 16 --image-size 276 368 --num-sample $NUM_SAMPLE
# 3:4
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x240p_3_4 \
    --num-frames 16 --image-size 368 276 --num-sample $NUM_SAMPLE
# 1:2
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x240p_1_2 \
    --num-frames 16 --image-size 226 452 --num-sample $NUM_SAMPLE
# 2:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x240p_2_1 \
    --num-frames 16 --image-size 452 226 --num-sample $NUM_SAMPLE

# == 64x240p ==
# 1:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x240p_1_1 \
    --num-frames 64 --image-size 320 320 --num-sample $NUM_SAMPLE
# 16:9
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x240p_16_9 \
    --num-frames 64 --image-size 240 426 --num-sample $NUM_SAMPLE
# 9:16
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x240p_9_16 \
    --num-frames 64 --image-size 426 240 --num-sample $NUM_SAMPLE
# 4:3
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x240p_4_3 \
    --num-frames 64 --image-size 276 368 --num-sample $NUM_SAMPLE
# 3:4
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x240p_3_4 \
    --num-frames 64 --image-size 368 276 --num-sample $NUM_SAMPLE
# 1:2
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x240p_1_2 \
    --num-frames 64 --image-size 226 452 --num-sample $NUM_SAMPLE
# 2:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x240p_2_1 \
    --num-frames 64 --image-size 452 226 --num-sample $NUM_SAMPLE

# == 128x240p ==
# 1:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 128x240p_1_1 \
    --num-frames 128 --image-size 320 320 --num-sample $NUM_SAMPLE
# 16:9
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 128x240p_16_9 \
    --num-frames 128 --image-size 240 426 --num-sample $NUM_SAMPLE
# 9:16
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 128x240p_9_16 \
    --num-frames 128 --image-size 426 240 --num-sample $NUM_SAMPLE
# 4:3
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 128x240p_4_3 \
    --num-frames 128 --image-size 276 368 --num-sample $NUM_SAMPLE
# 3:4
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 128x240p_3_4 \
    --num-frames 128 --image-size 368 276 --num-sample $NUM_SAMPLE
# 1:2
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 128x240p_1_2 \
    --num-frames 128 --image-size 226 452 --num-sample $NUM_SAMPLE
# 2:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 128x240p_2_1 \
    --num-frames 128 --image-size 452 226 --num-sample $NUM_SAMPLE

# == 16x360p ==
# 1:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x360p_1_1 \
    --num-frames 16 --image-size 480 480 --num-sample $NUM_SAMPLE
# 16:9
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x360p_16_9 \
    --num-frames 16 --image-size 360 640 --num-sample $NUM_SAMPLE
# 9:16
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x360p_9_16 \
    --num-frames 16 --image-size 640 360 --num-sample $NUM_SAMPLE
# 4:3
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x360p_4_3 \
    --num-frames 16 --image-size 416 554 --num-sample $NUM_SAMPLE
# 3:4
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x360p_3_4 \
    --num-frames 16 --image-size 554 416 --num-sample $NUM_SAMPLE
# 1:2
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x360p_1_2 \
    --num-frames 16 --image-size 360 640 --num-sample $NUM_SAMPLE
# 2:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x360p_2_1 \
    --num-frames 16 --image-size 640 360 --num-sample $NUM_SAMPLE

# == 64x360p ==
# 1:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x360p_1_1 \
    --num-frames 64 --image-size 480 480 --num-sample $NUM_SAMPLE
# 16:9
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x360p_16_9 \
    --num-frames 64 --image-size 360 640 --num-sample $NUM_SAMPLE
# 9:16
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x360p_9_16 \
    --num-frames 64 --image-size 640 360 --num-sample $NUM_SAMPLE
# 4:3
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x360p_4_3 \
    --num-frames 64 --image-size 416 554 --num-sample $NUM_SAMPLE
# 3:4
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x360p_3_4 \
    --num-frames 64 --image-size 554 416 --num-sample $NUM_SAMPLE
# 1:2
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x360p_1_2 \
    --num-frames 64 --image-size 360 640 --num-sample $NUM_SAMPLE
# 2:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x360p_2_1 \
    --num-frames 64 --image-size 640 360 --num-sample $NUM_SAMPLE

# == 128x360p ==
# 1:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 128x360p_1_1 \
    --num-frames 128 --image-size 480 480 --num-sample $NUM_SAMPLE
# 16:9
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 128x360p_16_9 \
    --num-frames 128 --image-size 360 640 --num-sample $NUM_SAMPLE
# 9:16
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 128x360p_9_16 \
    --num-frames 128 --image-size 640 360 --num-sample $NUM_SAMPLE
# 4:3
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 128x360p_4_3 \
    --num-frames 128 --image-size 416 554 --num-sample $NUM_SAMPLE
# 3:4
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 128x360p_3_4 \
    --num-frames 128 --image-size 554 416 --num-sample $NUM_SAMPLE
# 1:2
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 128x360p_1_2 \
    --num-frames 128 --image-size 360 640 --num-sample $NUM_SAMPLE
# 2:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 128x360p_2_1 \
    --num-frames 128 --image-size 640 360 --num-sample $NUM_SAMPLE

# == 16x480p ==
# 1:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x480p_1_1 \
    --num-frames 16 --image-size 640 640 --num-sample $NUM_SAMPLE
# 16:9
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x480p_16_9 \
    --num-frames 16 --image-size 480 854 --num-sample $NUM_SAMPLE
# 9:16
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x480p_9_16 \
    --num-frames 16 --image-size 854 480 --num-sample $NUM_SAMPLE
# 4:3
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x480p_4_3 \
    --num-frames 16 --image-size 554 738 --num-sample $NUM_SAMPLE
# 3:4
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x480p_3_4 \
    --num-frames 16 --image-size 738 554 --num-sample $NUM_SAMPLE
# 1:2
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x480p_1_2 \
    --num-frames 16 --image-size 452 904 --num-sample $NUM_SAMPLE
# 2:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x480p_2_1 \
    --num-frames 16 --image-size 904 452 --num-sample $NUM_SAMPLE

# == 32x480p ==
# 1:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x480p_1_1 \
    --num-frames 32 --image-size 640 640 --num-sample $NUM_SAMPLE
# 16:9
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x480p_16_9 \
    --num-frames 32 --image-size 480 854 --num-sample $NUM_SAMPLE
# 9:16
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x480p_9_16 \
    --num-frames 32 --image-size 854 480 --num-sample $NUM_SAMPLE
# 4:3
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x480p_4_3 \
    --num-frames 32 --image-size 554 738 --num-sample $NUM_SAMPLE
# 3:4
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x480p_3_4 \
    --num-frames 32 --image-size 738 554 --num-sample $NUM_SAMPLE
# 1:2
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x480p_1_2 \
    --num-frames 32 --image-size 452 904 --num-sample $NUM_SAMPLE
# 2:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x480p_2_1 \
    --num-frames 32 --image-size 904 452 --num-sample $NUM_SAMPLE

# == 64x480p ==
# 1:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x480p_1_1 \
    --num-frames 64 --image-size 640 640 --num-sample $NUM_SAMPLE
# 16:9
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x480p_16_9 \
    --num-frames 64 --image-size 480 854 --num-sample $NUM_SAMPLE
# 9:16
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x480p_9_16 \
    --num-frames 64 --image-size 854 480 --num-sample $NUM_SAMPLE
# 4:3
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x480p_4_3 \
    --num-frames 64 --image-size 554 738 --num-sample $NUM_SAMPLE
# 3:4
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x480p_3_4 \
    --num-frames 64 --image-size 738 554 --num-sample $NUM_SAMPLE
# 1:2
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x480p_1_2 \
    --num-frames 64 --image-size 452 904 --num-sample $NUM_SAMPLE
# 2:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 64x480p_2_1 \
    --num-frames 64 --image-size 904 452 --num-sample $NUM_SAMPLE

# == 16x720p ==
# 1:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x720p_1_1 \
    --num-frames 16 --image-size 960 960 --num-sample $NUM_SAMPLE
# 16:9
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x720p_16_9 \
    --num-frames 16 --image-size 720 1280 --num-sample $NUM_SAMPLE
# 9:16
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x720p_9_16 \
    --num-frames 16 --image-size 1280 720 --num-sample $NUM_SAMPLE
# 4:3
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x720p_4_3 \
    --num-frames 16 --image-size 832 1108 --num-sample $NUM_SAMPLE
# 3:4
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x720p_3_4 \
    --num-frames 16 --image-size 1108 832 --num-sample $NUM_SAMPLE
# 1:2
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x720p_1_2 \
    --num-frames 16 --image-size 1358 600 --num-sample $NUM_SAMPLE
# 2:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x720p_2_1 \
    --num-frames 16 --image-size 600 1358

# == 32x720p ==
# 1:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x720p_1_1 \
    --num-frames 32 --image-size 960 960 --num-sample $NUM_SAMPLE
# 16:9
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x720p_16_9 \
    --num-frames 32 --image-size 720 1280 --num-sample $NUM_SAMPLE
# 9:16
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x720p_9_16 \
    --num-frames 32 --image-size 1280 720 --num-sample $NUM_SAMPLE
# 4:3
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x720p_4_3 \
    --num-frames 32 --image-size 832 1108 --num-sample $NUM_SAMPLE
# 3:4
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x720p_3_4 \
    --num-frames 32 --image-size 1108 832 --num-sample $NUM_SAMPLE
# 1:2
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x720p_1_2 \
    --num-frames 32 --image-size 1358 600 --num-sample $NUM_SAMPLE
# 2:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x720p_2_1 \
    --num-frames 32 --image-size 600 1358

### End

end=$(date +%s)

runtime=$((end - start))

echo "Runtime: $runtime seconds"

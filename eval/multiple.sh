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
OUTPUT="./samples_${CKPT_BASE}_${NAME}"
start=$(date +%s)

# Generate samples

# 16x240p

# 64x240p

# 128x240p

# 16x320p

# 64x320p

# 128x320p

# 16x480p
# 1:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x480p_1_1 \
    --num-frames 16 --image-size 360 360 --num-samples $NUM_SAMPLE
# 16:9
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x480p_16_9 \
    --num-frames 16 --image-size 360 640 --num-samples $NUM_SAMPLE
# 9:16
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x480p_9_16 \
    --num-frames 16 --image-size 1280 720 --num-samples $NUM_SAMPLE
# 4:3
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x480p_4_3 \
    --num-frames 16 --image-size 832 1108 --num-samples $NUM_SAMPLE
# 3:4
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x480p_3_4 \
    --num-frames 16 --image-size 1108 832 --num-samples $NUM_SAMPLE
# 1:2
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x480p_1_2 \
    --num-frames 16 --image-size 1358 600 --num-samples $NUM_SAMPLE
# 2:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x480p_2_1 \
    --num-frames 16 --image-size 600 1358

# 32x480p

# 64x480p


# 16x720p
# 1:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x720p_1_1 \
    --num-frames 16 --image-size 960 960 --num-samples $NUM_SAMPLE
# 16:9
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x720p_16_9 \
    --num-frames 16 --image-size 720 1280 --num-samples $NUM_SAMPLE
# 9:16
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x720p_9_16 \
    --num-frames 16 --image-size 1280 720 --num-samples $NUM_SAMPLE
# 4:3
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x720p_4_3 \
    --num-frames 16 --image-size 832 1108 --num-samples $NUM_SAMPLE
# 3:4
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x720p_3_4 \
    --num-frames 16 --image-size 1108 832 --num-samples $NUM_SAMPLE
# 1:2
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x720p_1_2 \
    --num-frames 16 --image-size 1358 600 --num-samples $NUM_SAMPLE
# 2:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 16x720p_2_1 \
    --num-frames 16 --image-size 600 1358

# 32x720p
# 1:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x720p_1_1 \
    --num-frames 32 --image-size 960 960 --num-samples $NUM_SAMPLE
# 16:9
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x720p_16_9 \
    --num-frames 32 --image-size 720 1280 --num-samples $NUM_SAMPLE
# 9:16
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x720p_9_16 \
    --num-frames 32 --image-size 1280 720 --num-samples $NUM_SAMPLE
# 4:3
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x720p_4_3 \
    --num-frames 32 --image-size 832 1108 --num-samples $NUM_SAMPLE
# 3:4
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x720p_3_4 \
    --num-frames 32 --image-size 1108 832 --num-samples $NUM_SAMPLE
# 1:2
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x720p_1_2 \
    --num-frames 32 --image-size 1358 600 --num-samples $NUM_SAMPLE
# 2:1
eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --sample-name 32x720p_2_1 \
    --num-frames 32 --image-size 600 1358

### End

end=$(date +%s)

runtime=$((end - start))

echo "Runtime: $runtime seconds"

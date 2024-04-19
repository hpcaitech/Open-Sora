#!/bin/bash

set -x
set -e

CKPT=$1

CMD="python scripts/inference.py configs/opensora-v1-1/inference/sample.py"
CMD_REF="python scripts/inference-long.py configs/opensora-v1-1/inference/sample.py"
if [[ $CKPT == *"ema"* ]]; then
  parentdir=$(dirname $CKPT)
  CKPT_BASE=$(basename $parentdir)_ema
else
  CKPT_BASE=$(basename $CKPT)
fi
OUTPUT="./samples/samples_${CKPT_BASE}"
start=$(date +%s)

### Functions

function run_image() { # 10min
  # 1.1 1024x1024
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2i_samples.txt --save-dir $OUTPUT --num-frames 1 --image-size 1024 1024 --sample-name 1024x1024

  # 1.2 240x426
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2i_samples.txt --save-dir $OUTPUT --num-frames 1 --image-size 240 426 --sample-name 240x426 --end-index 3

  # 1.3 512x512
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2i_samples.txt --save-dir $OUTPUT --num-frames 1 --image-size 512 512 --sample-name 512x512 --end-index 3
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 1 --image-size 512 512 --sample-name 512x512 --end-index 3
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 1 --image-size 512 512 --sample-name 512x512 --end-index 3
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 1 --image-size 512 512 --sample-name 512x512 --end-index 3

  # 1.4 720p multi-resolution
  # 1:1
  PROMPT="Bright scene, aerial view,ancient city, fantasy, gorgeous light, mirror reflection, high detail, wide angle lens."
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 1 --image-size 960 960 --sample-name 720p_1_1
  # 16:9
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 1 --image-size 720 1280 --sample-name 720p_16_9
  # 9:16
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 1 --image-size 1280 720 --sample-name 720p_9_16
  # 4:3
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 1 --image-size 832 1108 --sample-name 720p_4_3
  # 3:4
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 1 --image-size 1108 832 --sample-name 720p_3_4
  # 1:2
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 1 --image-size 1358 600 --sample-name 720p_1_2
  # 2:1
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 1 --image-size 600 1358 --sample-name 720p_2_1
}

function run_video_1() { # 20min
  # 2.1.1 16x240x426
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 16 --image-size 240 426 --sample-name sample_16x240x426

  # 2.1.2 16x720p multi-resolution
  # 1:1
  PROMPT="A soaring drone footage captures the majestic beauty of a coastal cliff, its red and yellow stratified rock faces rich in color and against the vibrant turquoise of the sea. Seabirds can be seen taking flight around the cliff's precipices. As the drone slowly moves from different angles, the changing sunlight casts shifting shadows that highlight the rugged textures of the cliff and the surrounding calm sea. The water gently laps at the rock base and the greenery that clings to the top of the cliff, and the scene gives a sense of peaceful isolation at the fringes of the ocean. The video captures the essence of pristine natural beauty untouched by human structures."
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 16 --image-size 960 960 --sample-name 720p_1_1
  # 16:9
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 16 --image-size 720 1280 --sample-name 720p_16_9
  # 9:16
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 16 --image-size 1280 720 --sample-name 720p_9_16
  # 4:3
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 16 --image-size 832 1108 --sample-name 720p_4_3
  # 3:4
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 16 --image-size 1108 832 --sample-name 720p_3_4
  # 1:2
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 16 --image-size 1358 600 --sample-name 720p_1_2
  # 2:1
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 16 --image-size 600 1358 --sample-name 720p_2_1
}

function run_video_2() { # 60min
  # 2.2.1 16x240x426
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 16 --image-size 240 426 --sample-name short_16x240x426

  # 2.2.2 64x240x426
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 64 --image-size 240 426 --sample-name short_64x240x426
}

function run_video_3() { # 60min
  # 2.3.1 16x240x426
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 16 --image-size 240 426 --sample-name sora_16x240x426

  # 2.3.2 128x240x426
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 128 --image-size 240 426 --sample-name short_128x240x426
}

function run_video_4() { # 120min
  # 2.4 16x480x854
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 16 --image-size 480 854 --sample-name short_16x480x854
}

function run_video_5() { # 120min
  # 2.5 64x480x854
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 64 --image-size 480 854 --sample-name short_64x480x854
}

function run_video_6() { # 120min
  # 2.6 16x720x1280
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 16 --image-size 720 1280 --sample-name short_16x720x1280
}

function run_video_edit() { # 23min
  # 3.1 image-conditioned long video generation
  eval $CMD_REF --ckpt-path $CKPT --save-dir $OUTPUT --sample-name ref_L10C4_16x240x426 \
    --prompt-path assets/texts/t2v_ref.txt --start-index 0 --end-index 3 \
    --num-frames 16 --image-size 240 426 \
    --loop 5 --condition-frame-length 4 \
    --reference-path assets/images/condition/cliff.png assets/images/condition/wave.png assets/images/condition/ship.png \
    --mask-strategy "0,0,0,1,0" "0,0,0,1,0" "0,0,0,1,0"

  eval $CMD_REF --ckpt-path $CKPT --save-dir $OUTPUT --sample-name ref_L10C4_64x240x426 \
    --prompt-path assets/texts/t2v_ref.txt --start-index 0 --end-index 3 \
    --num-frames 64 --image-size 240 426 \
    --loop 5 --condition-frame-length 16 \
    --reference-path assets/images/condition/cliff.png assets/images/condition/wave.png assets/images/condition/ship.png \
    --mask-strategy "0,0,0,1,0" "0,0,0,1,0" "0,0,0,1,0"

  # 3.2
  eval $CMD_REF --ckpt-path $CKPT --save-dir $OUTPUT --sample-name ref_L1_128x240x426 \
    --prompt-path assets/texts/t2v_ref.txt --start-index 3 --end-index 5 \
    --num-frames 128 --image-size 240 426 \
    --loop 1 \
    --reference-path assets/images/condition/cliff.png "assets/images/condition/cactus-sad.png\;assets/images/condition/cactus-happy.png" \
    --mask-strategy "0,0,0,1,0\;0,0,0,1,-1" "0,0,0,1,0\;0,1,0,1,-1"
}

### Main

for arg in "$@"; do
  if [[ "$arg" = -1 ]] || [[ "$arg" = --image ]]; then
    echo "Running image samples..."
    run_image
  fi
  if [[ "$arg" = -2a ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples 1..."
    run_video_1
  fi
  if [[ "$arg" = -2b ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples 2..."
    run_video_2
  fi
  if [[ "$arg" = -2c ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples 3..."
    run_video_3
  fi
  if [[ "$arg" = -2d ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples 4..."
    run_video_4
  fi
  if [[ "$arg" = -2e ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples 5..."
    run_video_5
  fi
  if [[ "$arg" = -2f ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples 6..."
    run_video_6
  fi
  if [[ "$arg" = -3 ]] || [[ "$arg" = --video-edit ]]; then
    echo "Running video edit samples..."
    run_video_edit
  fi
done

### End

end=$(date +%s)

runtime=$((end - start))
echo "Runtime: $runtime seconds"

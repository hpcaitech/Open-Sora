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

function run_video_a() { # 20min
  # 2.1.1 16x240x426
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 16 --image-size 144 256 --sample-name sample_16x144x256
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 16 --image-size 256 256 --sample-name sample_16x256x256
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

function run_video_b() { # 30min
  # 2.2.1 16x240x426
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 16 --image-size 240 426 --sample-name short_16x240x426

  # 2.2.2 64x240x426
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 64 --image-size 240 426 --sample-name short_64x240x426
}

function run_video_c() { # 30min
  # 2.3.1 16x240x426
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 16 --image-size 240 426 --sample-name sora_16x240x426

  # 2.3.2 128x240x426
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 128 --image-size 240 426 --sample-name short_128x240x426
}

function run_video_d() { # 30min
  # 2.4 16x480x854
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 16 --image-size 480 854 --sample-name short_16x480x854
}

function run_video_e() { # 30min
  # 2.5 64x480x854
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 64 --image-size 480 854 --sample-name short_64x480x854
}

function run_video_f() { # 30min
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
    --prompt-path assets/texts/t2v_ref.txt --start-index 3 --end-index 6 \
    --num-frames 128 --image-size 240 426 \
    --loop 1 \
    --reference-path assets/images/condition/cliff.png "assets/images/condition/cactus-sad.png\;assets/images/condition/cactus-happy.png" https://cdn.openai.com/tmp/s/interp/d0.mp4 \
    --mask-strategy "0,0,0,1,0\;0,0,0,1,-1" "0,0,0,1,0\;0,1,0,1,-1" "0,0,0,64,0,0.5"
}

# vbench has 950 samples

function run_vbenck_a() { # 2h
  eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_dimension.txt \
    --start-index 0 --end-index 120
}

function run_vbenck_b() { # 2h
  eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_dimension.txt \
    --start-index 120 --end-index 240
}

function run_vbenck_c() { # 2h
  eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_dimension.txt \
    --start-index 240 --end-index 360
}

function run_vbenck_d() { # 2h
  eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_dimension.txt \
    --start-index 360 --end-index 480
}

function run_vbenck_e() { # 2h
  eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_dimension.txt \
    --start-index 480 --end-index 600
}

function run_vbenck_f() { # 2h
  eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_dimension.txt \
    --start-index 600 --end-index 720
}

function run_vbenck_g() { # 2h
  eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_dimension.txt \
    --start-index 720 --end-index 840
}

function run_vbenck_h() { # 2h
  eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_dimension.txt \
    --start-index 840
}

### Main

for arg in "$@"; do
  # image
  if [[ "$arg" = -1 ]] || [[ "$arg" = --image ]]; then
    echo "Running image samples..."
    run_image
  fi
  # video: sample 16x240p & multi-resolution
  if [[ "$arg" = -2a ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples a..."
    run_video_a
  fi
  # video: short 16x240p & 64x240p
  if [[ "$arg" = -2b ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples b..."
    run_video_b
  fi
  # video: sora 16x240p & short 128x240p
  if [[ "$arg" = -2c ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples c..."
    run_video_c
  fi
  # short 16x480p
  if [[ "$arg" = -2d ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples d..."
    run_video_d
  fi
  # short 64x480p
  if [[ "$arg" = -2e ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples e..."
    run_video_e
  fi
  # short 16x720p
  if [[ "$arg" = -2f ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples f..."
    run_video_f
  fi
  # video edit
  if [[ "$arg" = -3 ]] || [[ "$arg" = --video-edit ]]; then
    echo "Running video edit samples..."
    run_video_edit
  fi
  # vbench
  if [[ "$arg" = -4a ]] || [[ "$arg" = --vbench ]]; then
    echo "Running vbench samples a..."
    run_vbenck_a
  fi
  if [[ "$arg" = -4b ]] || [[ "$arg" = --vbench ]]; then
    echo "Running vbench samples b..."
    run_vbenck_b
  fi
  if [[ "$arg" = -4c ]] || [[ "$arg" = --vbench ]]; then
    echo "Running vbench samples c..."
    run_vbenck_c
  fi
  if [[ "$arg" = -4d ]] || [[ "$arg" = --vbench ]]; then
    echo "Running vbench samples d..."
    run_vbenck_d
  fi
  if [[ "$arg" = -4e ]] || [[ "$arg" = --vbench ]]; then
    echo "Running vbench samples e..."
    run_vbenck_e
  fi
  if [[ "$arg" = -4f ]] || [[ "$arg" = --vbench ]]; then
    echo "Running vbench samples f..."
    run_vbenck_f
  fi
  if [[ "$arg" = -4g ]] || [[ "$arg" = --vbench ]]; then
    echo "Running vbench samples g..."
    run_vbenck_g
  fi
  if [[ "$arg" = -4h ]] || [[ "$arg" = --vbench ]]; then
    echo "Running vbench samples h..."
    run_vbenck_h
  fi
done

### End

end=$(date +%s)

runtime=$((end - start))

echo "Runtime: $runtime seconds"

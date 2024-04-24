#!/bin/bash

# set -x
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
DEFAULT_BS=8

### Functions

function run_image() { # 10min
  # 1.1 1024x1024
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2i_samples.txt --save-dir $OUTPUT --num-frames 1 --image-size 1024 1024 --sample-name 1024x1024 --batch-size $DEFAULT_BS

  # 1.2 240x426
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2i_samples.txt --save-dir $OUTPUT --num-frames 1 --image-size 240 426 --sample-name 240x426 --end-index 3 --batch-size $DEFAULT_BS

  # 1.3 512x512
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2i_samples.txt --save-dir $OUTPUT --num-frames 1 --image-size 512 512 --sample-name 512x512 --end-index 3 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 1 --image-size 512 512 --sample-name 512x512 --end-index 3 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 1 --image-size 512 512 --sample-name 512x512 --end-index 3 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 1 --image-size 512 512 --sample-name 512x512 --end-index 3 --batch-size $DEFAULT_BS

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

function run_video_a() { # 30min, sample & multi-resolution
  # sample
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 16 --image-size 144 256 --sample-name sample_16x144x256 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 16 --image-size 240 426 --sample-name sample_16x240x426 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 32 --image-size 240 426 --sample-name sample_32x240x426 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 64 --image-size 240 426 --sample-name sample_64x240x426 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 16 --image-size 480 854 --sample-name sample_16x480x854 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 32 --image-size 480 854 --sample-name sample_32x480x854 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 16 --image-size 720 1280 --sample-name sample_16x720x1280 --batch-size $DEFAULT_BS
}

function run_video_b() { # 30min, short 16x240p & 64x240p
  # 32x240p, short
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 32 --image-size 240 426 --sample-name short_32x240x426 --batch-size $DEFAULT_BS

  # 64x240p, short
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 64 --image-size 240 426 --sample-name short_64x240x426 --batch-size $DEFAULT_BS
}

function run_video_c() { # 30min, sora 16x240p & short 128x240p
  # 16x240p, sora
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 16 --image-size 426 240 --sample-name sora_16x426x240 --batch-size $DEFAULT_BS

  # 16x240p, sora
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 16 --image-size 240 426 --sample-name sora_16x240x426 --batch-size $DEFAULT_BS

  # 128x240p, sora
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 128 --image-size 240 426 --sample-name sora_128x240x426 --batch-size $DEFAULT_BS
}

function run_video_d() { # 30min, sora 32x480p
  # 32x480p, short
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 32 --image-size 480 854 --sample-name short_32x480x854 --batch-size $DEFAULT_BS
}

function run_video_e() { # 30min
  # 64x480p, sora
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 64 --image-size 480 854 --sample-name sora_64x480x854 --batch-size 4
}

function run_video_f() { # 30min
  # 16x720p, sora
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 16 --image-size 720 1280 --sample-name sora_16x720x1280 --batch-size $DEFAULT_BS
}

function run_video_g() {
  # 16x720p multi-resolution
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

function run_video_h() { # 23min
  # 3.1 image-conditioned long video generation
  eval $CMD_REF --ckpt-path $CKPT --save-dir $OUTPUT --sample-name ref_L10C4_16x240x426 \
    --prompt-path assets/texts/t2v_ref.txt --start-index 0 --end-index 3 \
    --num-frames 16 --image-size 240 426 \
    --loop 5 --condition-frame-length 4 \
    --reference-path assets/images/condition/cliff.png assets/images/condition/wave.png assets/images/condition/ship.png \
    --mask-strategy "0" "0" "0" --batch-size $DEFAULT_BS

  eval $CMD_REF --ckpt-path $CKPT --save-dir $OUTPUT --sample-name ref_L10C4_64x240x426 \
    --prompt-path assets/texts/t2v_ref.txt --start-index 0 --end-index 3 \
    --num-frames 64 --image-size 240 426 \
    --loop 5 --condition-frame-length 16 \
    --reference-path assets/images/condition/cliff.png assets/images/condition/wave.png assets/images/condition/ship.png \
    --mask-strategy "0" "0" "0" --batch-size $DEFAULT_BS

  # 3.2
  eval $CMD_REF --ckpt-path $CKPT --save-dir $OUTPUT --sample-name ref_L1_128x240x426 \
    --prompt-path assets/texts/t2v_ref.txt --start-index 3 --end-index 6 \
    --num-frames 128 --image-size 240 426 \
    --loop 1 \
    --reference-path assets/images/condition/cliff.png "assets/images/condition/cactus-sad.png\;assets/images/condition/cactus-happy.png" https://cdn.openai.com/tmp/s/interp/d0.mp4 \
    --mask-strategy "0\;0,0,0,-1,1" "0\;0,1,0,-1,1" "0,0,0,0,64,0.5" --batch-size $DEFAULT_BS
}

# vbench has 950 samples

VBENCH_BS=32 # 80GB
VBENCH_FRAMES=16
VBENCH_H=240
VBENCH_W=426

function run_vbenck_a() { # 2h
  eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_dimension.txt \
    --batch-size $VBENCH_BS --num-frames $VBENCH_FRAMES --image-size $VBENCH_H $VBENCH_W --start-index 0 --end-index 120
}

function run_vbenck_b() { # 2h
  eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_dimension.txt --batch-size $VBENCH_BS --num-frames $VBENCH_FRAMES --image-size $VBENCH_H $VBENCH_W --start-index 120 --end-index 240
}

function run_vbenck_c() { # 2h
  eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_dimension.txt \
    --batch-size $VBENCH_BS --num-frames $VBENCH_FRAMES --image-size $VBENCH_H $VBENCH_W --start-index 240 --end-index 360
}

function run_vbenck_d() { # 2h
  eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_dimension.txt \
    --batch-size $VBENCH_BS --num-frames $VBENCH_FRAMES --image-size $VBENCH_H $VBENCH_W --start-index 360 --end-index 480
}

function run_vbenck_e() { # 2h
  eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_dimension.txt \
    --batch-size $VBENCH_BS --num-frames $VBENCH_FRAMES --image-size $VBENCH_H $VBENCH_W --start-index 480 --end-index 600
}

function run_vbenck_f() { # 2h
  eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_dimension.txt \
    --batch-size $VBENCH_BS --num-frames $VBENCH_FRAMES --image-size $VBENCH_H $VBENCH_W --start-index 600 --end-index 720
}

function run_vbenck_g() { # 2h
  eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_dimension.txt \
    --batch-size $VBENCH_BS --num-frames $VBENCH_FRAMES --image-size $VBENCH_H $VBENCH_W --start-index 720 --end-index 840
}

function run_vbenck_h() { # 2h
  eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_dimension.txt \
    --batch-size $VBENCH_BS --num-frames $VBENCH_FRAMES --image-size $VBENCH_H $VBENCH_W --start-index 840
}

# vbench-i2v has 1120 samples

VBENCH_I2V_FRAMES=16
VBENCH_I2V_H=256
VBENCH_I2V_W=256

function run_vbenck_i2v_a() {
  eval $CMD_REF --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_i2v.txt \
    --start-index 0 --end-index 140 \
    --num-frames $VBENCH_I2V_FRAMES --image-size $VBENCH_I2V_H $VBENCH_I2V_W --batch-size $VBENCH_BS
}

function run_vbenck_i2v_b() {
  eval $CMD_REF --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_i2v.txt \
    --start-index 140 --end-index 280 \
    --num-frames $VBENCH_I2V_FRAMES --image-size $VBENCH_I2V_H $VBENCH_I2V_W --batch-size $VBENCH_BS
}

function run_vbenck_i2v_c() {
  eval $CMD_REF --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_i2v.txt \
    --start-index 280 --end-index 420 \
    --num-frames $VBENCH_I2V_FRAMES --image-size $VBENCH_I2V_H $VBENCH_I2V_W --batch-size $VBENCH_BS
}

function run_vbenck_i2v_d() {
  eval $CMD_REF --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_i2v.txt \
    --start-index 420 --end-index 560 \
    --num-frames $VBENCH_I2V_FRAMES --image-size $VBENCH_I2V_H $VBENCH_I2V_W --batch-size $VBENCH_BS
}

function run_vbenck_i2v_e() {
  eval $CMD_REF --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_i2v.txt \
    --start-index 560 --end-index 700 \
    --num-frames $VBENCH_I2V_FRAMES --image-size $VBENCH_I2V_H $VBENCH_I2V_W --batch-size $VBENCH_BS
}

function run_vbenck_i2v_f() {
  eval $CMD_REF --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_i2v.txt \
    --start-index 700 --end-index 840 \
    --num-frames $VBENCH_I2V_FRAMES --image-size $VBENCH_I2V_H $VBENCH_I2V_W --batch-size $VBENCH_BS
}

function run_vbenck_i2v_g() {
  eval $CMD_REF --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_i2v.txt \
    --start-index 840 --end-index 980 \
    --num-frames $VBENCH_I2V_FRAMES --image-size $VBENCH_I2V_H $VBENCH_I2V_W --batch-size $VBENCH_BS
}

function run_vbenck_i2v_h() {
  eval $CMD_REF --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
    --prompt-path assets/texts/VBench/all_i2v.txt \
    --start-index 980 \
    --num-frames $VBENCH_I2V_FRAMES --image-size $VBENCH_I2V_H $VBENCH_I2V_W --batch-size $VBENCH_BS
}

### Main

for arg in "$@"; do
  # image
  if [[ "$arg" = -1 ]] || [[ "$arg" = --image ]]; then
    echo "Running image samples..."
    run_image
  fi
  if [[ "$arg" = -2a ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples a..."
    run_video_a
  fi
  if [[ "$arg" = -2b ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples b..."
    run_video_b
  fi
  if [[ "$arg" = -2c ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples c..."
    run_video_c
  fi
  if [[ "$arg" = -2d ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples d..."
    run_video_d
  fi
  if [[ "$arg" = -2e ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples e..."
    run_video_e
  fi
  if [[ "$arg" = -2f ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples f..."
    run_video_f
  fi
  if [[ "$arg" = -2g ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples g..."
    run_video_g
  fi
  if [[ "$arg" = -2h ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples h..."
    run_video_h
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
  # vbench-i2v
  if [[ "$arg" = -5a ]] || [[ "$arg" = --vbench-i2v ]]; then
    echo "Running vbench-i2v samples a..."
    run_vbenck_i2v_a
  fi
  if [[ "$arg" = -5b ]] || [[ "$arg" = --vbench-i2v ]]; then
    echo "Running vbench-i2v samples b..."
    run_vbenck_i2v_b
  fi
  if [[ "$arg" = -5c ]] || [[ "$arg" = --vbench-i2v ]]; then
    echo "Running vbench-i2v samples c..."
    run_vbenck_i2v_c
  fi
  if [[ "$arg" = -5d ]] || [[ "$arg" = --vbench-i2v ]]; then
    echo "Running vbench-i2v samples d..."
    run_vbenck_i2v_d
  fi
  if [[ "$arg" = -5e ]] || [[ "$arg" = --vbench-i2v ]]; then
    echo "Running vbench-i2v samples e..."
    run_vbenck_i2v_e
  fi
  if [[ "$arg" = -5f ]] || [[ "$arg" = --vbench-i2v ]]; then
    echo "Running vbench-i2v samples f..."
    run_vbenck_i2v_f
  fi
  if [[ "$arg" = -5g ]] || [[ "$arg" = --vbench-i2v ]]; then
    echo "Running vbench-i2v samples g..."
    run_vbenck_i2v_g
  fi
  if [[ "$arg" = -5h ]] || [[ "$arg" = --vbench-i2v ]]; then
    echo "Running vbench-i2v samples h..."
    run_vbenck_i2v_h
  fi
done

### End

end=$(date +%s)

runtime=$((end - start))

echo "Runtime: $runtime seconds"

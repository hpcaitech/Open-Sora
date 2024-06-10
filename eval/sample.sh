# !/bin/bash

CKPT=$1
NUM_FRAMES=$2
MODEL_NAME=$3

VBENCH_START_INDEX=$5
VBENCH_END_INDEX=$6
VBENCH_RES=$7
VBENCH_ASP_RATIO=$8

echo "NUM_FRAMES=${NUM_FRAMES}"

if [ -z "${NUM_FRAMES}" ]; then
  echo "you need to pass NUM_FRAMES"
else
  let DOUBLE_FRAMES=$2*2
  let QUAD_FRAMES=$2*4
  let OCT_FRAMES=$2*8
fi

echo "DOUBLE_FRAMES=${DOUBLE_FRAMES}"
echo "QUAD_FRAMES=${QUAD_FRAMES}"
echo "OCT_FRAMES=${OCT_FRAMES}"


CMD="python scripts/inference.py configs/opensora-v1-2/inference/sample.py"
if [[ $CKPT == *"ema"* ]]; then
  parentdir=$(dirname $CKPT)
  CKPT_BASE=$(basename $parentdir)_ema
else
  CKPT_BASE=$(basename $CKPT)
fi
OUTPUT="/mnt/jfs-hdd/sora/samples/samples_${MODEL_NAME}_${CKPT_BASE}"
start=$(date +%s)
DEFAULT_BS=1

### Functions

# called inside run_video_b
function run_image() { # 14min
  # # 1.1 1024x1024
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2i_samples.txt --save-dir $OUTPUT --num-frames 1 --image-size 1024 1024 --sample-name 1024x1024 --batch-size $DEFAULT_BS

  # 1.2 240x426
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2i_samples.txt --save-dir $OUTPUT --num-frames 1 --image-size 240 426 --sample-name 240x426 --end-index 3 --batch-size $DEFAULT_BS

  # 1.3 512x512
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2i_samples.txt --save-dir $OUTPUT --num-frames 1 --image-size 512 512 --sample-name t2i_512x512 --end-index 3 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 1 --image-size 512 512 --sample-name t2v_512x512 --end-index 3 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 1 --image-size 512 512 --sample-name short_512x512 --end-index 3 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 1 --image-size 512 512 --sample-name sora_512x512 --end-index 3 --batch-size $DEFAULT_BS

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

function run_video_a() { # 42min, sample & multi-resolution
  # sample
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames $NUM_FRAMES --image-size 144 256 --sample-name sample_${NUM_FRAMES}x144x256 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames $NUM_FRAMES --image-size 240 426 --sample-name sample_${NUM_FRAMES}x240x426 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames $DOUBLE_FRAMES --image-size 240 426 --sample-name sample_${DOUBLE_FRAMES}x240x426 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames $QUAD_FRAMES --image-size 240 426 --sample-name sample_${QUAD_FRAMES}x240x426 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames $NUM_FRAMES --image-size 480 854 --sample-name sample_${NUM_FRAMES}x480x854 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames $DOUBLE_FRAMES --image-size 480 854 --sample-name sample_${DOUBLE_FRAMES}x480x854 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames $NUM_FRAMES --image-size 720 1280 --sample-name sample_${NUM_FRAMES}x720x1280 --batch-size $DEFAULT_BS
}

function run_video_b() { # 18min + 14min = 32min, short 16x240p & 64x240p
  # run image
  echo "Inside run_video_b, running image samples..."
  run_image

  echo "Inside run_video_b, running video samples..."
  # 32x240p, short
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames $DOUBLE_FRAMES --image-size 240 426 --sample-name short_${DOUBLE_FRAMES}x240x426 --batch-size $DEFAULT_BS

  # 64x240p, short
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames $QUAD_FRAMES --image-size 240 426 --sample-name short_${QUAD_FRAMES}x240x426 --batch-size $DEFAULT_BS
}

function run_video_c() { # 60min, sora 16x240p & short 128x240p
  # 16x240p, sora
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames $NUM_FRAMES --image-size 426 240 --sample-name sora_${NUM_FRAMES}x426x240 --batch-size $DEFAULT_BS

  # 16x240p, sora
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames $NUM_FRAMES --image-size 240 426 --sample-name sora_${NUM_FRAMES}x240x426 --batch-size $DEFAULT_BS

  # 128x240p, sora
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames $OCT_FRAMES  --image-size 240 426 --sample-name sora_${OCT_FRAMES}x240x426 --batch-size $DEFAULT_BS
}

function run_video_d() { # 21min + 30min = 51min, sora 32x480p
  # 32x480p, short
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames $DOUBLE_FRAMES --image-size 480 854 --sample-name short_${DOUBLE_FRAMES}x480x854 --batch-size $DEFAULT_BS

  # 64x480p, sora, moved from run_video_e
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames $QUAD_FRAMES --image-size 480 854 --sample-name sora_${QUAD_FRAMES}x480x854 --batch-size $DEFAULT_BS --start-index 0 --end-index 16
}

function run_video_e() { # 90min * 2/3 = 60min
  # 64x480p, sora
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames $QUAD_FRAMES --image-size 480 854 --sample-name sora_${QUAD_FRAMES}x480x854 --batch-size $DEFAULT_BS --start-index 16 --end-index 100
}

function run_video_f() { # 60min
  # 16x720p, sora
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames $NUM_FRAMES --image-size 720 1280 --sample-name sora_${NUM_FRAMES}x720x1280 --batch-size $DEFAULT_BS
}

function run_video_g() { # 15min
  # 16x720p multi-resolution
  # 1:1
  PROMPT="A soaring drone footage captures the majestic beauty of a coastal cliff, its red and yellow stratified rock faces rich in color and against the vibrant turquoise of the sea. Seabirds can be seen taking flight around the cliff's precipices. As the drone slowly moves from different angles, the changing sunlight casts shifting shadows that highlight the rugged textures of the cliff and the surrounding calm sea. The water gently laps at the rock base and the greenery that clings to the top of the cliff, and the scene gives a sense of peaceful isolation at the fringes of the ocean. The video captures the essence of pristine natural beauty untouched by human structures."
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames $NUM_FRAMES --image-size 960 960 --sample-name 720p_1_1
  # 16:9
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames $NUM_FRAMES --image-size 720 1280 --sample-name 720p_16_9
  # 9:16
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames $NUM_FRAMES --image-size 1280 720 --sample-name 720p_9_16
  # 4:3
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames $NUM_FRAMES --image-size 832 1108 --sample-name 720p_4_3
  # 3:4
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames $NUM_FRAMES --image-size 1108 832 --sample-name 720p_3_4
  # 1:2
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames $NUM_FRAMES --image-size 1358 600 --sample-name 720p_1_2
  # 2:1
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames $NUM_FRAMES --image-size 600 1358 --sample-name 720p_2_1

  # add motion score
  eval $CMD --ckpt-path $CKPT --save-dir $OUTPUT --num-frames $NUM_FRAMES --resolution 720p --sample-name motion --prompt \
    \"A stylish woman walking in the street of Tokyo.\"\
    \"A stylish woman walking in the street of Tokyo. motion score: 0.0\" \
    \"A stylish woman walking in the street of Tokyo. motion score: 2.0\" \
    \"A stylish woman walking in the street of Tokyo. motion score: 4.0\" \
    \"A stylish woman walking in the street of Tokyo. motion score: 6.0\" \
    \"A stylish woman walking in the street of Tokyo. motion score: 10.0\" \
    \"A stylish woman walking in the street of Tokyo. motion score: 20.0\" \
    \"A stylish woman walking in the street of Tokyo. motion score: 30.0\" \
    \"A stylish woman walking in the street of Tokyo. motion score: 40.0\"

  # add aes score
  eval $CMD --ckpt-path $CKPT --save-dir $OUTPUT --num-frames $NUM_FRAMES --resolution 720p --sample-name aes --prompt \
    \"A stylish woman walking in the street of Tokyo.\"\
    \"A stylish woman walking in the street of Tokyo. aesthetic score: 4.0\" \
    \"A stylish woman walking in the street of Tokyo. aesthetic score: 4.5\" \
    \"A stylish woman walking in the street of Tokyo. aesthetic score: 5.0\" \
    \"A stylish woman walking in the street of Tokyo. aesthetic score: 5.5\" \
    \"A stylish woman walking in the street of Tokyo. aesthetic score: 6.0\" \
    \"A stylish woman walking in the street of Tokyo. aesthetic score: 6.5\" \
    \"A stylish woman walking in the street of Tokyo. aesthetic score: 7.0\"
}

function run_video_h() { # 61min
  # 3.1 image-conditioned long video generation
  eval $CMD --ckpt-path $CKPT --save-dir $OUTPUT --sample-name ref_L10C4_${NUM_FRAMES}x240x426 \
    --prompt-path assets/texts/t2v_ref.txt --start-index 0 --end-index 3 \
    --num-frames $NUM_FRAMES --image-size 240 426 \
    --loop 5 --condition-frame-length 15 \
    --reference-path assets/images/condition/cliff.png assets/images/condition/wave.png assets/images/condition/ship.png \
    --mask-strategy "0" "0" "0" --batch-size $DEFAULT_BS

  eval $CMD --ckpt-path $CKPT --save-dir $OUTPUT --sample-name ref_L10C4_${QUAD_FRAMES}x240x426 \
    --prompt-path assets/texts/t2v_ref.txt --start-index 0 --end-index 3 \
    --num-frames $QUAD_FRAMES --image-size 240 426 \
    --loop 5 --condition-frame-length 60 \
    --reference-path assets/images/condition/cliff.png assets/images/condition/wave.png assets/images/condition/ship.png \
    --mask-strategy "0" "0" "0" --batch-size $DEFAULT_BS

  # 3.2
  eval $CMD --ckpt-path $CKPT --save-dir $OUTPUT --sample-name ref_L1_${OCT_FRAMES}x240x426 \
    --prompt-path assets/texts/t2v_ref.txt --start-index 3 --end-index 6 \
    --num-frames $OCT_FRAMES  --image-size 240 426 \
    --loop 1 \
    --reference-path assets/images/condition/cliff.png "assets/images/condition/cactus-sad.png\;assets/images/condition/cactus-happy.png" https://cdn.openai.com/tmp/s/interp/d0.mp4 \
    --mask-strategy "0\;0,0,0,-1,1" "0\;0,1,0,-1,1" "0,0,0,0,${QUAD_FRAMES},0.5" --batch-size $DEFAULT_BS
}

# vbench has 950 samples

VBENCH_BS=1 # 80GB
VBENCH_H=240
VBENCH_W=426

function run_vbench() {
  if [ -z ${VBENCH_RES} ] || [ -z ${VBENCH_ASP_RATIO} ]  ;
      then
        eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
        --prompt-path assets/texts/VBench/all_dimension.txt \
        --image-size $VBENCH_H $VBENCH_W \
        --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
      else
        eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
        --prompt-path assets/texts/VBench/all_dimension.txt \
        --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO \
        --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
  fi
}

# vbench-i2v has 1120 samples

VBENCH_I2V_H=256
VBENCH_I2V_W=256

function run_vbench_i2v() {
    if [ -z ${VBENCH_RES} ] || [ -z ${VBENCH_ASP_RATIO} ]  ;
      then
        eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
        --prompt-path assets/texts/VBench/all_i2v.txt \
        --image-size $VBENCH_I2V_H $VBENCH_I2V_W \
        --start-index $1 --end-index $2 \
        --num-frames $NUM_FRAMES  --batch-size $VBENCH_BS
      else
        eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
        --prompt-path assets/texts/VBench/all_i2v.txt \
        --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO \
        --start-index $1 --end-index $2 \
        --num-frames $NUM_FRAMES --batch-size $VBENCH_BS
  fi

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
  if [[ "$arg" = -4 ]] || [[ "$arg" = --vbench ]]; then
    echo "Running vbench samples ..."
    if [ -z ${VBENCH_START_INDEX} ] || [ -z ${VBENCH_END_INDEX} ]  ;
      then
        echo "need to set start_index and end_index"
      else
          run_vbench $VBENCH_START_INDEX $VBENCH_END_INDEX
    fi
  fi
  # vbench-i2v
  if [[ "$arg" = -5 ]] || [[ "$arg" = --vbench-i2v ]]; then
    echo "Running vbench-i2v samples ..."
    if [ -z ${VBENCH_START_INDEX} ] || [ -z ${VBENCH_END_INDEX} ]  ;
      then
        echo "need to set start_index and end_index"
      else
          run_vbench_i2v $VBENCH_START_INDEX $VBENCH_END_INDEX
    fi
  fi
done

### End

end=$(date +%s)

runtime=$((end - start))

echo "Runtime: $runtime seconds"

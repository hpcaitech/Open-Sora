# !/bin/bash

CKPT=$1
NUM_FRAMES=$2
MODEL_NAME=$3
TASK_TYPE=$4
VBENCH_START_INDEX=$5
VBENCH_END_INDEX=$6
VBENCH_RES=$7
VBENCH_ASP_RATIO=$8

NUM_SAMPLING_STEPS=$9
FLOW=${10}
LLM_REFINE=${11}

BASE_ASPECT_RATIO=360p
ASPECT_RATIOS=(144p 240p 360p 480p 720p 1080p)
# Loop through the list of aspect ratios
i=0
for r in "${ASPECT_RATIOS[@]}"; do
  if [[ "$r" == "$BASE_ASPECT_RATIO" ]]; then
    # get aspect ratio 1 level up
    if [[ $((i+1)) -lt ${#ASPECT_RATIOS[@]} ]]; then
      ASPECT_RATIO_INCR_1=${ASPECT_RATIOS[$((i+1))]}
    else
      # If this is the highest ratio, return the highest ratio
      ASPECT_RATIO_INCR_1=${ASPECT_RATIOS[-1]}
    fi
    # get aspect ratio 2 levels up
    if [[ $((i+2)) -lt ${#ASPECT_RATIOS[@]} ]]; then
      ASPECT_RATIO_INCR_2=${ASPECT_RATIOS[$((i+2))]}
    else
      # If this is the highest ratio, return the highest ratio
      ASPECT_RATIO_INCR_2=${ASPECT_RATIOS[-1]}
    fi
  fi
  i=$((i+1))
done
echo "base aspect ratio: ${BASE_ASPECT_RATIO}"
echo "aspect ratio 1 level up: ${ASPECT_RATIO_INCR_1}"
echo "aspect ratio 2 levels up: ${ASPECT_RATIO_INCR_2}"
echo "Note that this aspect ratio level setting is used for videos only, not images"

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
  # 1.1 1024x1024
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2i_samples.txt --save-dir $OUTPUT --num-frames 1 --resolution 1024 --aspect-ratio 1:1 --sample-name image_1024_1_1 --batch-size $DEFAULT_BS

  # 1.2 240x426
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2i_samples.txt --save-dir $OUTPUT --num-frames 1 --resolution 240p --aspect-ratio 9:16 --sample-name image_240p_9_16 --end-index 3 --batch-size $DEFAULT_BS

  # 1.3 512x512
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2i_samples.txt --save-dir $OUTPUT --num-frames 1 --resolution 512 --aspect-ratio 1:1 --sample-name image_t2i_512_1_1 --end-index 3 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 1 --resolution 512 --aspect-ratio 1:1 --sample-name image_t2v_512_1_1 --end-index 3 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 1 --resolution 512 --aspect-ratio 1:1 --sample-name image_short_512_1_1 --end-index 3 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 1 --resolution 512 --aspect-ratio 1:1 --sample-name image_sora_512_1_1 --end-index 3 --batch-size $DEFAULT_BS

  # 1.4 720p multi-resolution
  # 1:1
  PROMPT="Bright scene, aerial view,ancient city, fantasy, gorgeous light, mirror reflection, high detail, wide angle lens."
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 1 --resolution 720p --aspect-ratio 1:1 --sample-name image_720p_1_1
  # 9:16
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 1 --resolution 720p --aspect-ratio 9:16 --sample-name image_720p_9_16
  # 16:9
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 1 --resolution 720p --aspect-ratio 16:9 --sample-name image_720p_16_9
  # 4:3
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 1 --resolution 720p --aspect-ratio 4:3 --sample-name image_720p_4_3
  # 3:4
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 1 --resolution 720p --aspect-ratio 3:4 --sample-name image_720p_3_4
  # 1:2
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 1 --resolution 720p --aspect-ratio 1:2 --sample-name image_720p_1_2
  # 2:1
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 1 --resolution 720p --aspect-ratio 2:1 --sample-name image_720p_2_1
}

# for (sample, short, sora)
#   for ( (4s, 720p), (8s, 480p), (16s, 360p) )

function run_video_a() { # ~ 30min ?
  ### previous cmds  # 42min, sample & multi-resolution
  # # sample, 144p, 9:16, 2s
  # eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 2s --resolution 144p --aspect-ratio 9:16 --sample-name sample_2s_144p_9_16 --batch-size $DEFAULT_BS
  # # sample, 240p, 9:16, 2s
  # eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 2s --resolution 240p --aspect-ratio 9:16 --sample-name sample_2s_240p_9_16 --batch-size $DEFAULT_BS
  # # sample, 240p, 9:16, 4s
  # eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 4s --resolution 240p --aspect-ratio 9:16 --sample-name sample_4s_240p_9_16 --batch-size $DEFAULT_BS
  # # sample, 240p, 9:16, 8s
  # eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 8s --resolution 240p --aspect-ratio 9:16 --sample-name sample_8s_240p_9_16 --batch-size $DEFAULT_BS
  # # sample, 480p, 9:16, 2s
  # eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 2s --resolution 480p --aspect-ratio 9:16 --sample-name sample_2s_480p_9_16 --batch-size $DEFAULT_BS
  # # sample, 480p, 9:16, 4s
  # eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 4s --resolution 480p --aspect-ratio 9:16 --sample-name sample_4s_480p_9_16 --batch-size $DEFAULT_BS
  # # sample, 720p, 9:16, 2s
  # eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 2s --resolution 720p --aspect-ratio 9:16 --sample-name sample_2s_720p_9_16 --batch-size $DEFAULT_BS

  # sample, 720p, 9:16, 2s
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 4s --resolution ${ASPECT_RATIO_INCR_2} --aspect-ratio 9:16 --sample-name sample_4s_${ASPECT_RATIO_INCR_2} --batch-size $DEFAULT_BS

  # sample, 480p, 9:16, 8s
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 8s --resolution ${ASPECT_RATIO_INCR_1} --aspect-ratio 9:16 --sample-name sample_8s_${ASPECT_RATIO_INCR_1} --batch-size $DEFAULT_BS

  # sample, 360p, 9:16, 16s
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 16s --resolution ${BASE_ASPECT_RATIO} --aspect-ratio 9:16 --sample-name sample_16s_${BASE_ASPECT_RATIO} --batch-size $DEFAULT_BS
}

function run_video_b() { # 18min + 14min = 32min, short 16x240p & 64x240p
  # run image, 14min
  echo "Inside run_video_b, running image samples..."
  run_image

  echo "Inside run_video_b, running video samples..."

  ### previous cmds, 18min
  # # short, 240p, 9:16, 4s
  # eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 4s --resolution 240p --aspect-ratio 9:16 --sample-name short_4s_240p_9_16 --batch-size $DEFAULT_BS
  # # short, 240p, 9:16, 8s
  # eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 8s --resolution 240p --aspect-ratio 9:16 --sample-name short_8s_240p_9_16 --batch-size $DEFAULT_BS

  # short, 480p, 9:16, 8s: ~24min
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 8s --resolution ${ASPECT_RATIO_INCR_1} --aspect-ratio 9:16 --sample-name short_8s_${ASPECT_RATIO_INCR_1} --batch-size $DEFAULT_BS

  # short, 360p, 9:16, 16s: ~24min
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 16s --resolution ${BASE_ASPECT_RATIO} --aspect-ratio 9:16 --sample-name short_16s_${BASE_ASPECT_RATIO} --batch-size $DEFAULT_BS

}

function run_video_c() {
  ### previous cmds, 60min
  # # sora, 240p, 16:9, 2s
  # eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 2s --resolution 240p --aspect-ratio 16:9 --sample-name sora_2s_240p_16_9 --batch-size $DEFAULT_BS
  # # sora, 240p, 9:16, 2s
  # eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 2s --resolution 240p --aspect-ratio 9:16 --sample-name sora_2s_240p_9_16 --batch-size $DEFAULT_BS
  # # sora, 240p, 9:16, 16s
  # eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 16s --resolution 240p --aspect-ratio 9:16 --sample-name sora_16s_240p_9_16 --batch-size $DEFAULT_BS

  # short, 720p, 9:16, 2s: ~9min
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 4s --resolution ${ASPECT_RATIO_INCR_2} --aspect-ratio 9:16 --sample-name short_4s_${ASPECT_RATIO_INCR_2} --batch-size $DEFAULT_BS

  # sora, 360p, 9:16, 16s: ~40min
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 16s --resolution ${BASE_ASPECT_RATIO} --aspect-ratio 9:16 --sample-name sora_16s_${BASE_ASPECT_RATIO} --batch-size $DEFAULT_BS
}

function run_video_d() {
  ### previous cmds, 21min + 30min = 51min
  # # short, 480p, 9:16, 4s: 21min
  # eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 4s --resolution 480p --aspect-ratio 9:16 --sample-name short_4s_480p_9_16 --batch-size $DEFAULT_BS
  # # sora, 480p, 9:16, 8s, 1/3 # moved from run_video_e, 30min
  # eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 8s --resolution 480p --aspect-ratio 9:16 --sample-name sora_8s_480p_9_16 --batch-size $DEFAULT_BS --start-index 0 --end-index 16

  # sora, 480p, 9:16, 8s, 1/3 # moved from run_video_e, 30min
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 8s --resolution ${ASPECT_RATIO_INCR_1} --aspect-ratio 9:16 --sample-name sora_8s_${ASPECT_RATIO_INCR_1} --batch-size $DEFAULT_BS --start-index 0 --end-index 16
}

function run_video_e() { # 90min * 2/3 = 60min
  # sora, 480p, 9:16, 8s, 2/3
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 8s --resolution ${ASPECT_RATIO_INCR_1} --aspect-ratio 9:16 --sample-name sora_8s_${ASPECT_RATIO_INCR_1} --batch-size $DEFAULT_BS --start-index 16 --end-index 100
}

function run_video_f() { # 60min
  # sora, 720p, 9:16, 2s
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 4s --resolution ${ASPECT_RATIO_INCR_2} --aspect-ratio 9:16 --sample-name sora_4s_${ASPECT_RATIO_INCR_2} --batch-size $DEFAULT_BS
}

# --resolution 720p --aspect-ratio [16:9, 9:16, ...]

function run_video_g() { # 15min
  # 720p, 2s multi-resolution
  # 1:1
  PROMPT="A soaring drone footage captures the majestic beauty of a coastal cliff, its red and yellow stratified rock faces rich in color and against the vibrant turquoise of the sea. Seabirds can be seen taking flight around the cliff's precipices. As the drone slowly moves from different angles, the changing sunlight casts shifting shadows that highlight the rugged textures of the cliff and the surrounding calm sea. The water gently laps at the rock base and the greenery that clings to the top of the cliff, and the scene gives a sense of peaceful isolation at the fringes of the ocean. The video captures the essence of pristine natural beauty untouched by human structures."
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 2s --resolution ${ASPECT_RATIO_INCR_2} --aspect-ratio 1:1 --sample-name drone_cliff_prompt_${ASPECT_RATIO_INCR_2}_2s_1_1
  # 16:9
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 2s --resolution ${ASPECT_RATIO_INCR_2} --aspect-ratio 16:9 --sample-name drone_cliff_prompt_${ASPECT_RATIO_INCR_2}_2s_16_9
  # 9:16
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 2s --resolution ${ASPECT_RATIO_INCR_2} --aspect-ratio 9:16 --sample-name drone_cliff_prompt_${ASPECT_RATIO_INCR_2}_2s_9_16
  # 4:3
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 2s --resolution ${ASPECT_RATIO_INCR_2} --aspect-ratio 4:3 --sample-name drone_cliff_prompt_${ASPECT_RATIO_INCR_2}_2s_4_3
  # 3:4
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 2s --resolution ${ASPECT_RATIO_INCR_2} --aspect-ratio 3:4 --sample-name drone_cliff_prompt_${ASPECT_RATIO_INCR_2}_2s_3_4
  # 1:2
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 2s --resolution ${ASPECT_RATIO_INCR_2} --aspect-ratio 1:2 --sample-name drone_cliff_prompt_${ASPECT_RATIO_INCR_2}_2s_1_2
  # 2:1
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 2s --resolution ${ASPECT_RATIO_INCR_2} --aspect-ratio 2:1 --sample-name drone_cliff_prompt_${ASPECT_RATIO_INCR_2}_2s_2_1

  # add motion score
  eval $CMD --ckpt-path $CKPT --save-dir $OUTPUT --num-frames 2s --resolution ${ASPECT_RATIO_INCR_2} --sample-name motion_2s_${ASPECT_RATIO_INCR_2} --prompt \
    \"A stylish woman walking in the street of Tokyo.\" \"A stylish woman walking in the street of Tokyo. motion score: 0.0\" \
    \"A stylish woman walking in the street of Tokyo. motion score: 2.0\" \
    \"A stylish woman walking in the street of Tokyo. motion score: 4.0\" \
    \"A stylish woman walking in the street of Tokyo. motion score: 6.0\" \
    \"A stylish woman walking in the street of Tokyo. motion score: 10.0\" \
    \"A stylish woman walking in the street of Tokyo. motion score: 25.0\" \
    \"A stylish woman walking in the street of Tokyo. motion score: 50.0\" \
    \"A stylish woman walking in the street of Tokyo. motion score: 100.0\"

  # add aes score
  eval $CMD --ckpt-path $CKPT --save-dir $OUTPUT --num-frames 2s --resolution ${ASPECT_RATIO_INCR_2} --sample-name aes_2s_${ASPECT_RATIO_INCR_2} --prompt \
    \"A stylish woman walking in the street of Tokyo.\" \"A stylish woman walking in the street of Tokyo. aesthetic score: 4.0\" \
    \"A stylish woman walking in the street of Tokyo. aesthetic score: 4.5\" \
    \"A stylish woman walking in the street of Tokyo. aesthetic score: 5.0\" \
    \"A stylish woman walking in the street of Tokyo. aesthetic score: 5.5\" \
    \"A stylish woman walking in the street of Tokyo. aesthetic score: 6.0\" \
    \"A stylish woman walking in the street of Tokyo. aesthetic score: 6.5\" \
    \"A stylish woman walking in the street of Tokyo. aesthetic score: 7.0\"
}

# resolution -> 480p

function run_video_h() { # 61min
  # 3.1 image-conditioned long video generation
  eval $CMD --ckpt-path $CKPT --save-dir $OUTPUT --sample-name ref_L5C5_2s_${BASE_ASPECT_RATIO}_9_16 \
    --prompt-path assets/texts/t2v_ref.txt --start-index 0 --end-index 3 \
    --num-frames 2s --resolution ${BASE_ASPECT_RATIO} --aspect-ratio 9:16 \
    --loop 5 --condition-frame-length 5 \
    --reference-path assets/images/condition/cliff.png assets/images/condition/wave.png assets/images/condition/ship.png \
    --mask-strategy "0" "0" "0" --batch-size $DEFAULT_BS

  eval $CMD --ckpt-path $CKPT --save-dir $OUTPUT --sample-name ref_L5C10_16s_${BASE_ASPECT_RATIO}_9_16 \
    --prompt-path assets/texts/t2v_ref.txt --start-index 0 --end-index 3 \
    --num-frames 16s --resolution ${BASE_ASPECT_RATIO} --aspect-ratio 9:16 \
    --loop 5 --condition-frame-length 10 \
    --reference-path assets/images/condition/cliff.png assets/images/condition/wave.png assets/images/condition/ship.png \
    --mask-strategy "0" "0" "0" --batch-size $DEFAULT_BS

  # 3.2
  eval $CMD --ckpt-path $CKPT --save-dir $OUTPUT --sample-name ref_L1_16s_${BASE_ASPECT_RATIO}_9_16 \
    --prompt-path assets/texts/t2v_ref.txt --start-index 3 --end-index 6 \
    --num-frames 16s --resolution ${BASE_ASPECT_RATIO} --aspect-ratio 9:16 \
    --loop 1 \
    --reference-path assets/images/condition/cliff.png "assets/images/condition/cactus-sad.png\;assets/images/condition/cactus-happy.png" https://cdn.openai.com/tmp/s/interp/d0.mp4 \
    --mask-strategy "0" "0\;0,1,0,-1,1" "0,0,0,0,${QUAD_FRAMES},0.5" --batch-size $DEFAULT_BS
}

# vbench has 950 samples

VBENCH_BS=1 # 80GB
VBENCH_H=240
VBENCH_W=426

function run_vbench() {
  if [ -z ${VBENCH_RES} ] || [ -z ${VBENCH_ASP_RATIO} ]; then
    eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
      --prompt-path assets/texts/VBench/all_dimension.txt \
      --image-size $VBENCH_H $VBENCH_W \
      --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
  else
    if [ -z ${NUM_SAMPLING_STEPS} ]; then
        eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
        --prompt-path assets/texts/VBench/all_dimension.txt \
        --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO \
        --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
    else
      if [ -z ${FLOW} ]; then
        eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
        --prompt-path assets/texts/VBench/all_dimension.txt \
        --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO --num-sampling-steps ${NUM_SAMPLING_STEPS} \
        --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
      else
        if [ -z ${LLM_REFINE} ]; then
          eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
          --prompt-path assets/texts/VBench/all_dimension.txt \
          --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO --num-sampling-steps ${NUM_SAMPLING_STEPS} --flow ${FLOW} \
          --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
        else
          if [ "${FLOW}" = "None" ]; then
            eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
            --prompt-path assets/texts/VBench/all_dimension.txt \
            --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO --num-sampling-steps ${NUM_SAMPLING_STEPS} --llm-refine ${LLM_REFINE} \
            --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
          else
            eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
            --prompt-path assets/texts/VBench/all_dimension.txt \
            --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO --num-sampling-steps ${NUM_SAMPLING_STEPS} --flow ${FLOW} --llm-refine ${LLM_REFINE} \
            --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
          fi
        fi
      fi
    fi
  fi
}

# vbench-i2v has 1120 samples

VBENCH_I2V_H=256
VBENCH_I2V_W=256

function run_vbench_i2v() {
  if [ -z ${VBENCH_RES} ] || [ -z ${VBENCH_ASP_RATIO} ]; then
    eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
      --prompt-path assets/texts/VBench/all_i2v.txt \
      --image-size $VBENCH_I2V_H $VBENCH_I2V_W \
      --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
  else
    if [ -z ${NUM_SAMPLING_STEPS} ]; then
        eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
        --prompt-path assets/texts/VBench/all_i2v.txt \
        --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO \
        --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
    else
      if [ -z ${FLOW} ]; then
        eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
        --prompt-path assets/texts/VBench/all_i2v.txt \
        --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO --num-sampling-steps ${NUM_SAMPLING_STEPS} \
        --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
      else
        if [ -z ${LLM_REFINE} ]; then
          eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
          --prompt-path assets/texts/VBench/all_i2v.txt \
          --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO --num-sampling-steps ${NUM_SAMPLING_STEPS} --flow ${FLOW} \
          --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
        else
          if [ "${FLOW}" = "None" ]; then
            eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
            --prompt-path assets/texts/VBench/all_i2v.txt \
            --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO --num-sampling-steps ${NUM_SAMPLING_STEPS} --llm-refine ${LLM_REFINE} \
            --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
          else
            eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
            --prompt-path assets/texts/VBench/all_i2v.txt \
            --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO --num-sampling-steps ${NUM_SAMPLING_STEPS} --flow ${FLOW} --llm-refine ${LLM_REFINE} \
            --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
          fi
        fi
      fi
    fi
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
    if [ -z ${VBENCH_START_INDEX} ] || [ -z ${VBENCH_END_INDEX} ]; then
      echo "need to set start_index and end_index"
    else
      run_vbench $VBENCH_START_INDEX $VBENCH_END_INDEX
    fi
  fi
  # vbench-i2v
  if [[ "$arg" = -5 ]] || [[ "$arg" = --vbench-i2v ]]; then
    echo "Running vbench-i2v samples ..."
    if [ -z ${VBENCH_START_INDEX} ] || [ -z ${VBENCH_END_INDEX} ]; then
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

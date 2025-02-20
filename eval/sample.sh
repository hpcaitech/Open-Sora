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
ASPECT_RATIOS=(360p 720p)
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

# CMD="python scripts/inference.py configs/opensora-v1-2/inference/sample.py"
CMD="python scripts/inference.py configs/opensora-v1-3/inference/t2v.py"
CMD_I2V="python scripts/inference_i2v.py configs/opensora-v1-3/inference/v2v.py"

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
function run_image() {
  # 360p multi-sample
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 1 --resolution 360p --aspect-ratio 1:1 --sample-name image_sora_360p_1_1 --end-index 3 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 1 --resolution 360p --aspect-ratio 1:1 --sample-name image_short_360p_1_1 --end-index 3 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 1 --resolution 360p --aspect-ratio 1:1 --sample-name image_t2v_360p_1_1 --end-index 3 --batch-size $DEFAULT_BS
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2i_samples.txt --save-dir $OUTPUT --num-frames 1 --resolution 360p --aspect-ratio 1:1 --sample-name image_t2i_360p_1_1 --end-index 3 --batch-size $DEFAULT_BS

  # 720p multi-resolution
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

function run_video_a() {
  # sample, 720p, 9:16
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 97 --resolution 720p --aspect-ratio 9:16 --sample-name sample_97_720p --batch-size $DEFAULT_BS

  # sample, 360p, 9:16
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 97 --resolution 360p --aspect-ratio 9:16 --sample-name sample_97_360p --batch-size $DEFAULT_BS

  # sample random type, 720p, 9:16
  if [[ -z "${OPENAI_API_KEY}" ]];
    then
      echo "Error: Required environment variable 'OPENAI_API_KEY' is not set."
      exit 1
    else
      eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/rand_types.txt --save-dir $OUTPUT --num-frames 97 --resolution 720p --aspect-ratio 9:16 --sample-name rand_types_2s_720p --batch-size $DEFAULT_BS --llm-refine True
  fi
}

function run_video_b() {
  echo "Inside run_video_b, running image samples..."
  run_image

  echo "Inside run_video_b, running video samples..."

  # short, 720p, 9:16
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 97 --resolution 720p --aspect-ratio 9:16 --sample-name short_97_720p --batch-size $DEFAULT_BS

  # short, 360p, 9:16
  eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 97 --resolution 360p --aspect-ratio 9:16 --sample-name short_97_360p --batch-size $DEFAULT_BS
}

function run_video_c() {
  # 720p, multi-resolution
  # 1:1
  PROMPT="A soaring drone footage captures the majestic beauty of a coastal cliff, its red and yellow stratified rock faces rich in color and against the vibrant turquoise of the sea. Seabirds can be seen taking flight around the cliff's precipices. As the drone slowly moves from different angles, the changing sunlight casts shifting shadows that highlight the rugged textures of the cliff and the surrounding calm sea. The water gently laps at the rock base and the greenery that clings to the top of the cliff, and the scene gives a sense of peaceful isolation at the fringes of the ocean. The video captures the essence of pristine natural beauty untouched by human structures."
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 49 --resolution 720p --aspect-ratio 1:1 --sample-name drone_cliff_prompt_720p_49_1_1
  # 16:9
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 49 --resolution 720p --aspect-ratio 16:9 --sample-name drone_cliff_prompt_720p_49_16_9
  # 9:16
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 49 --resolution 720p --aspect-ratio 9:16 --sample-name drone_cliff_prompt_720p_49_9_16
  # 4:3
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 49 --resolution 720p --aspect-ratio 4:3 --sample-name drone_cliff_prompt_720p_49_4_3
  # 3:4
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 49 --resolution 720p --aspect-ratio 3:4 --sample-name drone_cliff_prompt_720p_49_3_4
  # 1:2
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 49 --resolution 720p --aspect-ratio 1:2 --sample-name drone_cliff_prompt_720p_49_1_2
  # 2:1
  eval $CMD --ckpt-path $CKPT --prompt \"$PROMPT\" --save-dir $OUTPUT --num-frames 49 --resolution 720p --aspect-ratio 2:1 --sample-name drone_cliff_prompt_720p_49_2_1

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

# vbench has 950 samples

VBENCH_BS=1
VBENCH_H=360
VBENCH_W=640

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
        --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO \
        --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
      else
        if [ -z ${LLM_REFINE} ]; then
          eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
          --prompt-path assets/texts/VBench/all_dimension.txt \
          --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO --flow ${FLOW} \
          --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
        else
          if [ "${FLOW}" = "None" ]; then
            eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
            --prompt-path assets/texts/VBench/all_dimension.txt \
            --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO --llm-refine ${LLM_REFINE} \
            --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
          else
            eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --prompt-as-path --num-sample 5 \
            --prompt-path assets/texts/VBench/all_dimension.txt \
            --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATI --flow ${FLOW} --llm-refine ${LLM_REFINE} \
            --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
          fi
        fi
      fi
    fi
  fi
}

# vbench-i2v has 1120 samples

VBENCH_I2V_H=360
VBENCH_I2V_W=360

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
        --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO \
        --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
      else
        if [ -z ${LLM_REFINE} ]; then
          eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
          --prompt-path assets/texts/VBench/all_i2v.txt \
          --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO --flow ${FLOW} \
          --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
        else
          if [ "${FLOW}" = "None" ]; then
            eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
            --prompt-path assets/texts/VBench/all_i2v.txt \
            --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO --llm-refine ${LLM_REFINE} \
            --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
          else
            eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench_i2v --prompt-as-path --num-sample 5 \
            --prompt-path assets/texts/VBench/all_i2v.txt \
            --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO --flow ${FLOW} --llm-refine ${LLM_REFINE} \
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

BASE_MODEL_PATH=$1
TRAINED_MODEL_PATH=$2
I2V_HEAD_PATH=$3
I2V_TAIL_PATH=$4
I2V_LOOP_PATH=$5
I2V_ORI_PATH=$6


if [ -z $I2V_ORI_PATH ]; then
    I2V_ORI_PATH="assets/texts/i2v/prompts_ori.txt"
fi

if [ -z $I2V_HEAD_PATH ]; then
    I2V_HEAD_PATH="assets/texts/i2v/prompts_head.txt"
fi

if [ -z $I2V_TAIL_PATH ]; then
    I2V_TAIL_PATH="assets/texts/i2v/prompts_tail.txt"
fi

if [ -z $I2V_LOOP_PATH ]; then
    I2V_LOOP_PATH="assets/texts/i2v/prompts_loop.txt"
fi

STEP_RECORD=$(basename $TRAINED_MODEL_PATH)
if [ -z $SAVE_DIR ]; then
    SAVE_DIR="samples/i2v/test/${STEP_RECORD}"
fi
echo "save dir: ${SAVE_DIR}"

if [ -z $NUM_FRAMES ]; then
    NUM_FRAMES=49
fi
echo "num frames: ${NUM_FRAMES}"


command="python scripts/inference_i2v.py configs/opensora-v1-3/inference/v2v.py"

# # original uncond
# ${command} --ckpt-path ${BASE_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_ORI_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name base_uncond --use-sdedit False

# # trained uncond
# ${command} --ckpt-path ${TRAINED_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_ORI_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name trained_uncond --use-sdedit False

# trained uncond
# ${command} --ckpt-path ${TRAINED_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_ORI_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name trained_cond_none_image1osci --use-sdedit False --use-oscillation-guidance-for-image True --image-cfg-scale 1 --cond-type "none" --start-index 1 --end-index 2

# ${command} --ckpt-path ${TRAINED_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_ORI_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name trained_cond_none_image1osci_bias0 --use-sdedit False --use-oscillation-guidance-for-image True --image-cfg-scale 1 --cond-type "none" --start-index 0 --end-index 2


# trained cond: i2v_head
# ${command} --ckpt-path ${TRAINED_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_HEAD_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name trained_i2v_head_image1.5osci_text7.5osci --use-sdedit False --cond-type i2v_head --use-oscillation-guidance-for-image True --image-cfg-scale 1.5 --use-oscillation-guidance-for-text True --cfg-scale 7.5
# ${command} --ckpt-path ${TRAINED_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_HEAD_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name trained_i2v_head_image1.5osci_text7.5osci_bias0 --use-sdedit False --cond-type i2v_head --use-oscillation-guidance-for-image True --image-cfg-scale 1.5 --use-oscillation-guidance-for-text True --cfg-scale 7.5


# trained cond: i2v_tail
# ${command} --ckpt-path ${TRAINED_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_TAIL_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name trained_i2v_tail_image1.5osci_text7.5osci --use-sdedit False --cond-type i2v_tail --use-oscillation-guidance-for-image True --image-cfg-scale 1.5 --use-oscillation-guidance-for-text True --cfg-scale 7.5

# trained cond: i2v_head
# ${command} --ckpt-path ${TRAINED_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_HEAD_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name trained_i2v_head_image2osci_text7.5osci --use-sdedit False --cond-type i2v_head --use-oscillation-guidance-for-image True --image-cfg-scale 2 --use-oscillation-guidance-for-text True --cfg-scale 7.5

# trained cond: i2v_tail
# ${command} --ckpt-path ${TRAINED_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_TAIL_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name trained_i2v_tail_image2osci_text7.5osci --use-sdedit False --cond-type i2v_tail --use-oscillation-guidance-for-image True --image-cfg-scale 2 --use-oscillation-guidance-for-text True --cfg-scale 7.5

# trained cond: i2v_head
# ${command} --ckpt-path ${TRAINED_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_HEAD_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name trained_i2v_head_image2.5osci_text7.5osci --use-sdedit False --cond-type i2v_head --use-oscillation-guidance-for-image True --image-cfg-scale 2.5 --use-oscillation-guidance-for-text True --cfg-scale 7.5

# trained cond: i2v_tail
# ${command} --ckpt-path ${TRAINED_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_TAIL_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name trained_i2v_tail_image2.5osci_text7.5osci --use-sdedit False --cond-type i2v_tail --use-oscillation-guidance-for-image True --image-cfg-scale 2.5 --use-oscillation-guidance-for-text True --cfg-scale 7.5



# trained cond: i2v_loop
# ${command} --ckpt-path ${TRAINED_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_LOOP_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name trained_i2v_loop --use-sdedit False --cond-type i2v_loop --loop 2

# # traind cond: i2v_loop
# ${command} --ckpt-path ${TRAINED_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_LOOP_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name trained_i2v_loop_image1osci_text7.5osci --use-sdedit False --cond-type i2v_loop --use-oscillation-guidance-for-image True --image-cfg-scale 1 --use-oscillation-guidance-for-text True --cfg-scale 7.5

# # traind cond: i2v_loop
# ${command} --ckpt-path ${TRAINED_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_LOOP_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name trained_i2v_loop_image2osci_text7.5osci --use-sdedit False --cond-type i2v_loop --use-oscillation-guidance-for-image True --image-cfg-scale 2 --use-oscillation-guidance-for-text True --cfg-scale 7.5

# # traind cond: i2v_loop
# ${command} --ckpt-path ${TRAINED_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_LOOP_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name trained_i2v_loop_image3osci_text7.5osci --use-sdedit False --cond-type i2v_loop --use-oscillation-guidance-for-image True --image-cfg-scale 3 --use-oscillation-guidance-for-text True --cfg-scale 7.5

# # traind cond: i2v_loop, cfg text osci
# ${command} --ckpt-path ${TRAINED_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_LOOP_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name trained_i2v_loop_text7osci --use-sdedit False --cond-type i2v_loop --use-oscillation-guidance-for-text True --cfg-scale 7

# # trained cond: i2v_loop, image text osci
# ${command} --ckpt-path ${TRAINED_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_LOOP_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name trained_i2v_loop_text3.5osci_image3.5osci --use-sdedit False --cond-type i2v_loop --use-oscillation-guidance-for-text True --cfg-scale 3.5 --use-oscillation-guidance-for-image Tru

# # trained cond: i2v_loop, image text oscie --image-cfg-scale 3.5
# ${command} --ckpt-path ${TRAINED_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_LOOP_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name trained_i2v_loop_text7osci_image3.5osci --use-sdedit False --cond-type i2v_loop --use-oscillation-guidance-for-text True --cfg-scale 7 --use-oscillation-guidance-for-image True --image-cfg-scale 3.5






# # base cond: i2v_head
# ${command} --ckpt-path ${BASE_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_HEAD_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name base_i2v_head --use-sdedit False --cond-type i2v_head

# # base cond: i2v_tail
# ${command} --ckpt-path ${BASE_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_TAIL_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name base_i2v_tail --use-sdedit False --cond-type i2v_tail

# # base cond: i2v_loop
# ${command} --ckpt-path ${BASE_MODEL_PATH} --save-dir ${SAVE_DIR} --prompt-path ${I2V_LOOP_PATH} --num-frames ${NUM_FRAMES} --resolution 360p --aspect-ratio 9:16 --sample-name base_i2v_loop --use-sdedit False --cond-type i2v_loop

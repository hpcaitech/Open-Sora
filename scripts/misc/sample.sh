set -x;

CUDA_VISIBLE_DEVICES=7
CMD="python scripts/inference.py configs/opensora-v1-1/inference/sample.py"
CKPT="~/lishenggui/epoch0-global_step8500"
OUTPUT="./outputs/samples_s1_8500"

# 1. image
# 1.1 1024x1024
eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2i_samples.txt --save-dir $OUTPUT --num-frames 1 --image-size 1024 1024 --sample-name pixart_1024x1024_1

# 1.2 512x512

# 1.3 240x426

# 1.4 720p multi-resolution

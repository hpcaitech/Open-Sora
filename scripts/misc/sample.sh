set -x

CUDA_VISIBLE_DEVICES=7
CMD="python scripts/inference.py configs/opensora-v1-1/inference/sample.py"
CKPT="~/lishenggui/epoch0-global_step9000"
OUTPUT="./outputs/samples_s1_9000"

# 1. image

# 1.1 1024x1024
eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2i_samples.txt --save-dir $OUTPUT --num-frames 1 --image-size 1024 1024 --sample-name 1024x1024

# 1.2 240x426
eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2i_samples.txt --save-dir $OUTPUT --num-frames 1 --image-size 240 426 --sample-name 240x426 --end-idx 3

# 1.3 512x512
eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2i_samples.txt --save-dir $OUTPUT --num-frames 1 --image-size 512 512 --sample-name 512x512 --end-idx 3

# 1.4 720p multi-resolution
# 1:1
PROMPT = "Bright scene, aerial view,ancient city, fantasy, gorgeous light, mirror reflection, high detail, wide angle lens."
eval $CMD --ckpt-path $CKPT --prompt $PROMPT --save-dir $OUTPUT --num-frames 1 --image-size 960 960 --sample-name 720p_1_1
# 16:9
eval $CMD --ckpt-path $CKPT --prompt $PROMPT --save-dir $OUTPUT --num-frames 1 --image-size 720 1280 --sample-name 720p_16_9
# 9:16
eval $CMD --ckpt-path $CKPT --prompt $PROMPT --save-dir $OUTPUT --num-frames 1 --image-size 1280 720 --sample-name 720p_9_16
# 4:3
eval $CMD --ckpt-path $CKPT --prompt $PROMPT --save-dir $OUTPUT --num-frames 1 --image-size 832 1108 --sample-name 720p_4_3
# 3:4
eval $CMD --ckpt-path $CKPT --prompt $PROMPT --save-dir $OUTPUT --num-frames 1 --image-size 1108 832 --sample-name 720p_3_4
# 1:2
eval $CMD --ckpt-path $CKPT --prompt $PROMPT --save-dir $OUTPUT --num-frames 1 --image-size 1358 600 --sample-name 720p_1_2
# 2:1
eval $CMD --ckpt-path $CKPT --prompt $PROMPT --save-dir $OUTPUT --num-frames 1 --image-size 600 1358 --sample-name 720p_2_1

# 2. video

# 1.1 16x240x426
eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_samples.txt --save-dir $OUTPUT --num-frames 16 --image-size 240 426 --sample-name sample_16x240x426
eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 16 --image-size 240 426 --sample-name short_16x240x426
eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_sora.txt --save-dir $OUTPUT --num-frames 16 --image-size 240 426 --sample-name sora_16x240x426

# 1.2 64x240x426
eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 64 --image-size 240 426 --sample-name short_64x240x426

# 1.3 128x240x426
eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 128 --image-size 240 426 --sample-name short_128x240x426

# 1.4 16x480x854
eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 16 --image-size 480 854 --sample-name short_16x480x854

# 1.5 64x480x854
eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 64 --image-size 480 854 --sample-name short_64x480x854

# 1.6 16x720x1280
eval $CMD --ckpt-path $CKPT --prompt-path assets/texts/t2v_short.txt --save-dir $OUTPUT --num-frames 16 --image-size 720 1280 --sample-name short_16x720x1280

# 1.7 16x720p multi-resolution
# 1:1
PROMPT = "A soaring drone footage captures the majestic beauty of a coastal cliff, its red and yellow stratified rock faces rich in color and against the vibrant turquoise of the sea. Seabirds can be seen taking flight around the cliff's precipices. As the drone slowly moves from different angles, the changing sunlight casts shifting shadows that highlight the rugged textures of the cliff and the surrounding calm sea. The water gently laps at the rock base and the greenery that clings to the top of the cliff, and the scene gives a sense of peaceful isolation at the fringes of the ocean. The video captures the essence of pristine natural beauty untouched by human structures."
eval $CMD --ckpt-path $CKPT --prompt $PROMPT --save-dir $OUTPUT --num-frames 16 --image-size 960 960 --sample-name 720p_1_1
# 16:9
eval $CMD --ckpt-path $CKPT --prompt $PROMPT --save-dir $OUTPUT --num-frames 16 --image-size 720 1280 --sample-name 720p_16_9
# 9:16
eval $CMD --ckpt-path $CKPT --prompt $PROMPT --save-dir $OUTPUT --num-frames 16 --image-size 1280 720 --sample-name 720p_9_16
# 4:3
eval $CMD --ckpt-path $CKPT --prompt $PROMPT --save-dir $OUTPUT --num-frames 16 --image-size 832 1108 --sample-name 720p_4_3
# 3:4
eval $CMD --ckpt-path $CKPT --prompt $PROMPT --save-dir $OUTPUT --num-frames 16 --image-size 1108 832 --sample-name 720p_3_4
# 1:2
eval $CMD --ckpt-path $CKPT --prompt $PROMPT --save-dir $OUTPUT --num-frames 16 --image-size 1358 600 --sample-name 720p_1_2
# 2:1
eval $CMD --ckpt-path $CKPT --prompt $PROMPT --save-dir $OUTPUT --num-frames 16 --image-size 600 1358 --sample-name 720p_2_1

# 3. video edit

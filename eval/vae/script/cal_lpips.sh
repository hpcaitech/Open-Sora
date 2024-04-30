python eval_common_metric.py \
    --real_video_dir path/to/imageA\
    --generated_video_dir path/to/imageB \
    --batch_size 10 \
    --num_frames 20 \
    --crop_size 64 \
    --device 'cuda' \
    --metric 'lpips'
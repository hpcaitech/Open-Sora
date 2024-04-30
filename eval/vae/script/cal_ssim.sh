python eval_common_metric.py \
    --real_video_dir /data/xiaogeng_liu/data/video1 \
    --generated_video_dir /data/xiaogeng_liu/data/video2 \
    --batch_size 10 \
    --num_frames 20 \
    --crop_size 64 \
    --device 'cuda' \
    --metric 'ssim'
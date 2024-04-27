python eval_common_metric.py \
    --real_video_dir path/to/imageA\
    --generated_video_dir path/to/imageB \
    --batch_size 10 \
    --crop_size 64 \
    --num_frames 20 \
    --device 'cuda' \
    --metric 'fvd' \
    --fvd_method 'styleganv'

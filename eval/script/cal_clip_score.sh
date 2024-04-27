# clip_score cross modality
python eval_clip_score.py \
    --real_path path/to/image \
    --generated_path path/to/text \
    --batch-size 50 \
    --device "cuda"

# clip_score within the same modality
python eval_clip_score.py \
    --real_path path/to/textA \
    --generated_path path/to/textB \
    --real_flag txt \
    --generated_flag txt \
    --batch-size 50 \
    --device "cuda"

python eval_clip_score.py \
    --real_path path/to/imageA \
    --generated_path path/to/imageB \
    --real_flag img \
    --generated_flag img \
    --batch-size 50 \
    --device "cuda"

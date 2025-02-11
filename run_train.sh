export TENSORNVME_DEBUG=1
# one node
torchrun --standalone --nproc_per_node 4 scripts/train_benchmark.py \
    configs/opensora-v1-2/train/test_stage1_offline.py --data-path /home/xuyongsheng/working/dataset/mx_data/meta_clips_caption_cleaned.csv

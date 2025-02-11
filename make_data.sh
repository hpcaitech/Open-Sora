export TENSORNVME_DEBUG=1
# one node
torchrun --standalone --nproc_per_node 1 scripts/make_feat.py \
    configs/opensora-v1-2/train/test_stage1.py --data-path /home/xuyongsheng/working/dataset/mx_data/meta_clips_caption_cleaned.csv

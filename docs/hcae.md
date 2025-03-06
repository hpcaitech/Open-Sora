# Visit the high compression video autoencoder

## Introduction

## Traini

torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py configs/diffusion/inference/high_compression.py --dataset.data-path assets/texts/sora.csv --ckpt-path /mnt/jfs-hdd/sora/checkpoints/zhengzangwei/outputs/adapt_video_sana/250304_141109-diffusion_train_dc_ae_video_temporal_compression_i2v/epoch0-global_step2000 --save-dir samples/debug_03_06/adapt_2k_op --sampling_option.num_frames 128



torchrun --nproc_per_node 8 scripts/diffusion/train.py configs/diffusion/train/high_compression.py --dataset.data-path /mnt/ddn/sora/meta/vo3/stage2/video+image_stage2_nopart3.parquet --wandb True --wandb-project adapt_video_sana --load /mnt/jfs-hdd/sora/checkpoints/zhengzangwei/outputs/adapt_video_sana/250304_141109-diffusion_train_dc_ae_video_temporal_compression_i2v/epoch0-global_step2000 --outputs /mnt/jfs-hdd/sora/checkpoints/zhengzangwei/outputs/adapt_video_sana/ --start-step 0 --start-epoch 0 --seed 2026

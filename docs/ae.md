# Step by step to train and evaluate an video autoencoder

## Installation

```
pip install diffusers==0.31.0
```

## Training
The command to launch training is as follows:
```
torchrun --nproc_per_node 8 scripts/vae/train_video_sana.py configs/vae/train/video_dc_ae_train_channel_proj.py --dataset.data-path /home/zhengzangwei/Open-Sora/datasets/pexels_45k_necessary.csv --model.model_name dc-ae-f32t4c128 --wandb True --optim.lr 5e-5 --wandb-project dcae --vae_loss_config.perceptual_loss_weight 0.5 --wandb-expr-name dc-ae-f32t4c64_full_model_ft

```

## Inference

## Config Interpretation

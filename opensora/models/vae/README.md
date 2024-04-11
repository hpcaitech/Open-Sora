## Commands 


## 1. VAE 3D
### 1.1 Train

```yaml
# train on pexel dataset
WANDB_API_KEY=<wandb_api_key> CUDA_VISIBLE_DEVICES=<n> torchrun --master_port=<port_num> --nnodes=1 --nproc_per_node=1 scripts/train-vae.py configs/vae_3d/train/16x256x256.py --data-path /home/shenchenhui/data/pexels/train.csv --wandb True
```

### 1.2 Inference 

```yaml
CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/inference-vae.py configs/vae_3d/inference/16x256x256.py --ckpt-path /home/shenchenhui/Open-Sora-dev/outputs/train_pexel_028/epoch3-global_step20000/ --data-path /home/shenchenhui/data/pexels/debug.csv --save-dir outputs/pexel


# resume training debug
CUDA_VISIBLE_DEVICES=5 torchrun --master_port=29530 --nnodes=1 --nproc_per_node=1 scripts/train-vae.py configs/vae_3d/train/16x256x256.py --data-path /home/shenchenhui/data/pexels/debug.csv  --load /home/shenchenhui/Open-Sora-dev/outputs/006-F16S3-VAE_3D_B/epoch49-global_step50
```

## 2. MAGVIT-v2

### 2.1 dependencies
```
'accelerate>=0.24.0',
'beartype',
'einops>=0.7.0',
'ema-pytorch>=0.2.4',
'pytorch-warmup',
'gateloop-transformer>=0.2.2',
'kornia',
'opencv-python',
'pillow',
'pytorch-custom-utils>=0.0.9',
'numpy',
'vector-quantize-pytorch>=1.11.8',
'taylor-series-linear-attention>=0.1.5',
'torch',
'torchvision',
'x-transformers'
```



## Commands 

## 0. References

* https://github.com/google-research/magvit
* https://github.com/CompVis/taming-transformers
* https://github.com/adobe/antialiased-cnns/pull/39/commits/3d6f02b6943c58b68c19c07bc26fad57492ff3bc
* https://github.com/PKU-YuanGroup/Open-Sora-Plan


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

version 2 pipeline
```yaml
# NOTE: first VAE is pretrained 2D, 16x128x128 --> 16x16x16
# then we train our own temporal VAE, 16x16x16 --> 4x16x16
# we use a 3 layer discriminator on the intermediate of 16x16x16
WANDB_API_KEY=<wandb_api_key> CUDA_VISIBLE_DEVICES=7 torchrun --master_port=29580 --nnodes=1 --nproc_per_node=1 scripts/train-vae-v2.py configs/vae_magvit_v2/train/pipeline_16x128x128.py --data-path /home/shenchenhui/data/trial_data/train_short.csv --wandb True
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

Note: 
uses `hotfix/zero` branch of `https://github.com/ver217/ColossalAI.git`.
clone the repo, go to the branch, then do `pip install .` 


### 2.2 Train

```yaml
CUDA_VISIBLE_DEVICES7 torchrun --master_port=29510 --nnodes=1 --nproc_per_node=1 scripts/train-vae-v2.py configs/vae_magvit_v2/train/17x128x128.py --data-path /home/shenchenhui/data/pexels/train.csv
```

### 2.3 Inference


### 2.4 Data

full data combining the follwing: `/home/shenchenhui/data/pixabay+pexels.csv`

* ~/data/pixabay: `/home/data/sora_data/pixabay/raw/data/split-0`
* pexels: `/home/litianyi/data/pexels/processed/meta/pexels_caption_vinfo_ready_noempty_clean.csv`
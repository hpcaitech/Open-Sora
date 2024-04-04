## Commands 


### 1. Train

```yaml
# train on pexel dataset
WANDB_API_KEY=<wandb_api_key> CUDA_VISIBLE_DEVICES=<n> torchrun --master_port=<port_num> --nnodes=1 --nproc_per_node=1 scripts/train-vae.py configs/vae_3d/train/16x256x256.py --data-path /home/shenchenhui/data/pexels/train.csv --wandb True
```

### 2. Inference 

```yaml
CUDA_VISIBLE_DEVICES=<n> torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/inference-vae.py configs/vae_3d/inference/16x256x256.py --ckpt-path /home/shenchenhui/Open-Sora-dev/outputs/028-F16S3-VAE_3D/epoch3-global_step20000/vae --vae_only True --data-path /home/shenchenhui/data/pexels/test.csv
```
## Commands 


### 1. Train

```yaml
# train on pexel dataset
WANDB_API_KEY=<wandb_api_key> CUDA_VISIBLE_DEVICES=<n> torchrun --master_port=<port_num> --nnodes=1 --nproc_per_node=1 scripts/train-vae.py configs/vae_3d/train/16x256x256.py --data-path /home/shenchenhui/data/pexels/train.csv --wandb True
```

### 2. Inference 

```yaml
CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/inference-vae.py configs/vae_3d/inference/16x256x256.py --ckpt-path /home/shenchenhui/Open-Sora-dev/outputs/train_pexel_028/epoch3-global_step20000/ --data-path /home/shenchenhui/data/pexels/debug.csv --save-dir outputs/pexel

CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/inference-vae.py configs/vae_3d/inference/16x256x256.py --ckpt-path /home/shenchenhui/Open-Sora-dev/outputs/004-F16S3-VAE_3D_B/epoch0-global_step1000 --data-path /home/shenchenhui/data/pexels/debug.csv --save-dir outputs/pexel

CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/inference-vae.py configs/vae_3d/inference/16x256x256.py --ckpt-path /home/shenchenhui/Open-Sora-dev/outputs/004-F16S3-VAE_3D_B/epoch0-global_step2000 --data-path /home/shenchenhui/data/pexels/debug.csv --save-dir outputs/pexel


# debug train and inference on the same few samples
WANDB_API_KEY=7bc1ce71b2dc0b8cd40c500eb256747583f6c07e CUDA_VISIBLE_DEVICES=5 torchrun --master_port=29530 --nnodes=1 --nproc_per_node=1 scripts/train-vae.py configs/vae_3d/train/16x256x256.py --data-path /home/shenchenhui/data/pexels/debug.csv --wandb True

CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/inference-debug.py configs/vae_3d/inference/16x256x256.py --ckpt-path /home/shenchenhui/Open-Sora-dev/outputs/006-F16S3-VAE_3D_B/epoch49-global_step50 --data-path /home/shenchenhui/data/pexels/debug.csv

# resume training debug
WANDB_API_KEY=7bc1ce71b2dc0b8cd40c500eb256747583f6c07e CUDA_VISIBLE_DEVICES=5 torchrun --master_port=29530 --nnodes=1 --nproc_per_node=1 scripts/train-vae.py configs/vae_3d/train/16x256x256.py --data-path /home/shenchenhui/data/pexels/debug.csv  --load /home/shenchenhui/Open-Sora-dev/outputs/006-F16S3-VAE_3D_B/epoch49-global_step50 --wandb True 
```

```yaml
scp  -P 31081 shenchenhui@211.102.192.108:/home/shenchenhui/Open-Sora-dev/outputs/samples/sample_0.mp4 /Users/shenchenhui/Desktop

scp  -P 31081 shenchenhui@211.102.192.108:/home/shenchenhui/data/pexels/test.csv /Users/shenchenhui/Desktop


``` 
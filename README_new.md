## Quick Start

Text to Video via Image:

256px

768px

Image to Video:

256px

```python
```

768px

```python
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --standalone scripts/diffusion/inference.py configs/diffusion/inference/768px.py --ckpt-path /mnt/jfs-hdd/sora/release/vo2_1_768px_i2v.pt --save-dir samples/debug_03_04/t2v --prompt "raining, sea"
```

Text to Video:

256px

```python
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --standalone scripts/diffusion/inference.py configs/diffusion/inference/256px.py --ckpt-path /mnt/jfs-hdd/sora/release/vo2_1_768px_i2v.pt --save-dir samples/debug_03_03/t2v --prompt "raining, sea"
```

768px


```python
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --standalone scripts/diffusion/inference.py configs/diffusion/inference/768px.py --ckpt-path /mnt/jfs-hdd/sora/release/vo2_1_768px_i2v.pt --save-dir samples/debug_03_03/t2v --prompt "raining, sea"
```

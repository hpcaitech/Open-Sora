# Visit the high compression video autoencoder

## Introduction

## Inference

```bash
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py configs/diffusion/inference/high_compression.py --dataset.data-path assets/texts/sora.csv
```

## Training

```bash
torchrun --nproc_per_node 8 scripts/diffusion/train.py configs/diffusion/train/high_compression.py --dataset.data-path
```

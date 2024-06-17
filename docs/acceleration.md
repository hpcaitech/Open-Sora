# Acceleration

>This document corresponds to our v1.1 release

Open-Sora aims to provide a high-speed training framework for diffusion models. We can achieve **55%** training speed acceleration when training on **64 frames 512x512 videos**. Our framework support training **1min 1080p videos**.

## Accelerated Transformer

Open-Sora boosts the training speed by:

- Kernel optimization including [flash attention](https://github.com/Dao-AILab/flash-attention), fused layernorm kernel, and the ones compiled by colossalAI.
- Hybrid parallelism including ZeRO.
- Gradient checkpointing for larger batch size.

Our training speed on images is comparable to [OpenDiT](https://github.com/NUS-HPC-AI-Lab/OpenDiT), a project to accelerate DiT training. The training speed is measured on 8 H800 GPUs with batch size 128, image size 256x256.

| Model    | Throughput (img/s/GPU) | Throughput (tokens/s/GPU) |
| -------- | ---------------------- | ------------------------- |
| DiT      | 100                    | 26k                       |
| OpenDiT  | 175                    | 45k                       |
| OpenSora | 175                    | 45k                       |

## Efficient STDiT

Our STDiT adopts spatial-temporal attention to model the video data. Compared with directly applying full attention on DiT, our STDiT is more efficient as the number of frames increases. Our current framework only supports sequence parallelism for very long sequence.

The training speed is measured on 8 H800 GPUs with acceleration techniques applied, GC means gradient checkpointing. Both with T5 conditioning like PixArt.

| Model            | Setting        | Throughput (sample/s/GPU) | Throughput (tokens/s/GPU) |
| ---------------- | -------------- | ------------------------- | ------------------------- |
| DiT              | 16x256  (4k)   | 7.20                      | 29k                       |
| STDiT            | 16x256  (4k)   | 7.00                      | 28k                       |
| DiT              | 16x512  (16k)  | 0.85                      | 14k                       |
| STDiT            | 16x512  (16k)  | 1.45                      | 23k                       |
| DiT (GC)         | 64x512  (65k)  | 0.08                      | 5k                        |
| STDiT (GC)       | 64x512  (65k)  | 0.40                      | 25k                       |
| STDiT (GC, sp=2) | 360x512 (370k) | 0.10                      | 18k                       |

With a 4x downsampling in the temporal dimension with Video-VAE, an 24fps video has 450 frames. The gap between the speed of STDiT (28k tokens/s) and DiT on images (up to 45k tokens/s) mainly comes from the T5 and VAE encoding, and temporal attention.

## Accelerated Encoder (T5, VAE)

During training, texts are encoded by T5, and videos are encoded by VAE. Typically there are two ways to accelerate the training:

1. Preprocess text and video data in advance and save them to disk.
2. Encode text and video data during training, and accelerate the encoding process.

For option 1, 120 tokens for one sample require 1M disk space, and a 64x64x64 latent requires 4M. Considering a training dataset with 10M video clips, the total disk space required is 50TB. Our storage system is not ready at this time for this scale of data.

For option 2, we boost T5 speed and memory requirement. According to [OpenDiT](https://github.com/NUS-HPC-AI-Lab/OpenDiT), we find VAE consumes a large number of GPU memory. Thus we split batch size into smaller ones for VAE encoding. With both techniques, we can greatly accelerate the training speed.

The training speed is measured on 8 H800 GPUs with STDiT.

| Acceleration | Setting       | Throughput (img/s/GPU) | Throughput (tokens/s/GPU) |
| ------------ | ------------- | ---------------------- | ------------------------- |
| Baseline     | 16x256  (4k)  | 6.16                   | 25k                       |
| w. faster T5 | 16x256  (4k)  | 7.00                   | 29k                       |
| Baseline     | 64x512  (65k) | 0.94                   | 15k                       |
| w. both      | 64x512  (65k) | 1.45                   | 23k                       |

# Acceleration

## Accelerated Transformer

Open-Sora boosts the training speed by:

- Kernal optimization including [flash attention](https://github.com/Dao-AILab/flash-attention), fused layernorm kernal.
- Hybrid parallelism including ZeRO.

Our training speed on images is comparable to [OpenDiT](https://github.com/NUS-HPC-AI-Lab/OpenDiT), an project to accelerate DiT training. The training speed is measured on 8 H800 GPUs with batch size 128, image size 256x256.

| Model    | Throughput (img/s) |
| -------- | ------------------ |
| DiT      |                    |
| OpenDiT  |                    |
| OpenSora |                    |

## Efficient STDiT

Our STDiT adopts spatial-temporal attention to model the video data. Compared with directly applying full attention on DiT, our STDiT is more efficient as the number of frames increases. Our current framework only supports sequence parallelism for very long sequence.

The training speed is measured on 8 H800 GPUs with acceleration techniques applied.

| Model | Setting        | BS  | Throughput (img/s) | Throughput (tokens/s) |
| ----- | -------------- | --- | ------------------ | --------------------- |
| DiT   | 16x256  (4k)   | 64  | 
| STDiT | 16x256  (4k)   | 64  |
| DiT   | 16x512  (16k)  |
| STDiT | 16x512  (16k)  |
| DiT   | 64x512  (65k)  |
| STDiT | 64x512  (65k)  |
| DiT   | 256x512 (262k) |
| STDiT | 256x512 (262k) |

## Accelerated Encoder (T5, VAE)

During training, texts are encoded by T5, and videos are encoded by VAE. Typically there are two ways to accelerate the training:

1. Preprocess text and video data in advance and save them to disk.
2. Encode text and video data during training, and accelerate the encoding process.

For option 1, 120 tokens for one sample require 1M disk space, and a 64x64x64 latent requires 4M. Considering a training dataset with 10M video clips, the total disk space required is 50TB. Our storage system is not ready at this time for this scale of data.

For option 2, we boost T5 speed and memory requirement. According to [OpenDiT](https://github.com/NUS-HPC-AI-Lab/OpenDiT), we find VAE consumes a large number of GPU memory. Thus we split batch size into smaller ones for VAE encoding. With both techniques, we can greatly accelerated the training speed.

The training speed is measured on 8 H800 GPUs with STDiT.

| Acceleration | Setting | #Tokens | Throughput (img/s) | Throughput (tokens/s) |
| ------------ | ------- | ------- | ------------------ | --------------------- |
| Baseline     |
| w. faster T5 |
| Baseline     |
| w. faster T5 |
| w. both      |

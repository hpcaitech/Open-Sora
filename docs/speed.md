# Speed

## Training

The speeds are measured when training on a single node with 8 GPUs (H800-80GB) if not specified. The batch size (BS) is the local batch size.

For image (256x256), with 1x2x2 stride, one image equals 1x256x256 / 1x8x8 / 1x2x2 = 256 tokens.

| Model                 | BS  | Speed    | Speed (bs=32) |
| --------------------- | --- | -------- | ------------- |
| DiT-XL/2              | 128 | 1.3 it/s | 5.2 it/s      |
| PixArt-XL/2           | 64  | 1.2 it/s | 2.4 it/s      |
| PixArt-XL/2 (t5_null) | 128 | 1.1 it/s | 4.4 it/s      |

For video (16x256x256), one video equals 16x256x256 / 1x8x8 / 1x2x2 = 4096 tokens.

| Model          | BS  | Speed      | Speed (bs=8) |
| -------------- | --- | ---------- | ------------ |
| DiT-XL/2       | 8   | 1 it/s     | 1 it/s       |
| DiT-XL/2x2     | 20  | 0.625 it/s | 1.56 it/s    |
| Latte-XL/2     | 8   | 1.3 it/s   | 1.3 it/s     |
| Latte-XL/2x2   | 20  | 0.65 it/s  | 1.625 it/s   |
| DiT-ST-XL/2    | 8   | 1 it/s     | 1 it/s       |
| PixArt-XL/2    | 8   | 0.9 it/s   | 0.9 it/s     |
| PixArt-ST-XL/2 | 6   | 1.05 it/s  | 0.79 it/s    |

For different video resolution settings:

| Model                | Setting    | #Tokens | BS  | Speed     | Speed (bs=8) |
| -------------------- | ---------- | ------- | --- | --------- | ------------ |
| PixArt-XL/2          | 16x256x256 | 4096    | 8   | 0.90 it/s | 0.90 it/s    |
| PixArt-XL/2          | 16x512x512 | 16384   | 1   | 0.85 it/s | 0.10 it/s    |
| PixArt-ST-XL/2       | 16x256x256 | 4096    | 6   | 1.05 it/s | 0.79 it/s    |
| PixArt-ST-XL/2       | 16x512x512 | 16384   | 1   | 1.45 it/s | 0.18 it/s    |
| PixArt-ST-XL/2 w. GC | 16x256x256 | 4096    | 8   | 0.87 it/s | 1.16 it/s    |

## Inference

To be updated.

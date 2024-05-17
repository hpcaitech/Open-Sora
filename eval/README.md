# Evalution

## Human evaluation

To conduct human evaluation, we need to generate various samples. We provide many prompts in `assets/texts`, and defined some test setting covering different resolution, duration and aspect ratio in `eval/sample.sh`. To facilitate the usage of multiple GPUs, we split sampling tasks into several parts.

```bash
# image (1)
bash eval/sample.sh /path/to/ckpt num_frames model_name_for_log -1
# video (2a 2b 2c ...)
bash eval/sample.sh /path/to/ckpt num_frames model_name_for_log -2a
# launch 8 jobs at once (you must read the script to understand the details)
bash eval/human_eval/launch.sh /path/to/ckpt /path/to/ckpt num_frames model_name_for_log
```

## Rectified Flow Loss

Evaluate the rectified flow loss with the following commands.

```bash
# image
torchrun --standalone --nproc_per_node 1 eval/loss/eval_loss.py configs/opensora-v1-2/misc/eval_loss.py --data-path /path/to/img.csv --ckpt-path /path/to/ckpt

# video
torchrun --standalone --nproc_per_node 1 eval/loss/eval_loss.py configs/opensora-v1-2/misc/eval_loss.py --data-path /path/to/vid.csv --ckpt-path /path/to/ckpt

# select resolution
torchrun --standalone --nproc_per_node 1 eval/loss/eval_loss.py configs/opensora-v1-2/misc/eval_loss.py --data-path /path/to/vid.csv --ckpt-path /path/to/ckpt --resolution 720p
```

To launch multiple jobs at once, use the following script.

```bash
bash eval/loss/launch.sh /path/to/ckpt
```

## VBench

[VBench](https://github.com/Vchitect/VBench) is a benchmark for short text to video generation. We provide a script for easily generating samples required by VBench.

First, generate the relevant videos with the following commands:

```bash
# vbench tasks (4a 4b 4c ...)
bash eval/sample.sh /path/to/ckpt  -4a
# launch 8 jobs at once (you must read the script to understand the details)
bash eval/vbench/launch.sh /path/to/ckpt
```

After generation, install the VBench package following our [installation](../docs/installation.md)'s sections of "Evaluation Dependencies". Then, run the following commands to evaluate the generated samples.

```bash
bash eval/vbench/vbench.sh /path/to/video_folder
```

## VBench-i2v

[VBench-i2v](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v) is a benchmark for short image to video generation (beta version).

TBD

## VAE

Install the dependencies package following our [installation](../docs/installation.md)'s s sections of "Evaluation Dependencies". Then, run the following evaluation command:

```bash
# metric can any one or list of: ssim, psnr, lpips, flolpips
python eval/vae/eval_common_metric.py --batch_size 2 --real_video_dir path/to/original/videos --generated_video_dir path/to/generated/videos --device cuda --sample_fps 24 --crop_size 256 --resolution 256 --num_frames 17 --sample_rate 1 --metric ssim psnr lpips flolpips
```

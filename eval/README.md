# Evalution

## Human evaluation

To conduct human evaluation, we need to generate various samples. We provide many prompts in `assets/texts`, and defined some test setting covering different resolution, duration and aspect ratio in `eval/sample.sh`. To facilitate the usage of multiple GPUs, we split sampling tasks into several parts.

```bash
# image (1)
bash eval/sample.sh /path/to/ckpt num_frames model_name_for_log -1
# video (2a 2b 2c ...)
bash eval/sample.sh /path/to/ckpt num_frames model_name_for_log -2a
# launch 8 jobs at once (you must read the script to understand the details)
bash eval/human_eval/launch.sh /path/to/ckpt num_frames model_name_for_log
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
bash eval/loss/launch.sh /path/to/ckpt model_name
```

To obtain an organized list of scores:
```bash
python eval/loss/tabulate_rl_loss.py --log_dir path/to/log/dir
```

## VBench

[VBench](https://github.com/Vchitect/VBench) is a benchmark for short text to video generation. We provide a script for easily generating samples required by VBench.

First, generate the relevant videos with the following commands:

```bash
# vbench task, if evaluation all set start_index to 0, end_index to 2000
bash eval/sample.sh /path/to/ckpt num_frames model_name_for_log  -4 start_index end_index

# Alternatively, launch 8 jobs at once (you must read the script to understand the details)
bash eval/vbench/launch.sh /path/to/ckpt num_frames model_name

# in addition, you can specify resolution, aspect ratio, sampling steps, flow, and llm-refine
bash eval/vbench/launch.sh /path/to/ckpt num_frames model_name res_value aspect_ratio_value steps_value flow_value llm_refine_value
# for example
# bash eval/vbench/launch.sh /mnt/jfs-hdd/sora/checkpoints/outputs/042-STDiT3-XL-2/epoch1-global_step16200_llm_refine/ema.pt 51 042-STDiT3-XL-2 240p 9:16 30 2 True
```

After generation, install the VBench package following our [installation](../docs/installation.md)'s sections of "Evaluation Dependencies". Then, run the following commands to evaluate the generated samples.

<!-- ```bash
bash eval/vbench/vbench.sh /path/to/video_folder /path/to/model/ckpt
``` -->

```bash
python eval/vbench/calc_vbench.py /path/to/video_folder /path/to/model/ckpt
```

Finally, we obtain the scaled scores for the model by:
```bash
python eval/vbench/tabulate_vbench_scores.py --score_dir path/to/score/dir
```

## VBench-i2v

[VBench-i2v](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v) is a benchmark for short image to video generation (beta version).
Similarly, install the VBench package following our [installation](../docs/installation.md)'s sections of "Evaluation Dependencies".

```bash
# Step 1: generate the relevant videos
# vbench i2v tasks, if evaluation all set start_index to 0, end_index to 2000
bash eval/sample.sh /path/to/ckpt num_frames model_name_for_log -5 start_index end_index
# Alternatively, launch 8 jobs at once
bash eval/vbench_i2v/launch.sh /path/to/ckpt num_frames model_name

# Step 2: run vbench to evaluate the generated samples
python eval/vbench_i2v/vbench_i2v.py /path/to/video_folder /path/to/model/ckpt
# Note that if you need to go to `VBench/vbench2_beta_i2v/utils.py` and change the harded-coded var `image_root` in the `load_i2v_dimension_info` function to your corresponding image folder.

# Step 3: obtain the scaled scores
python eval/vbench_i2v/tabulate_vbench_i2v_scores.py path/to/videos/folder path/to/your/model/ckpt
# this will store the results under `eval/vbench_i2v` in the path/to/your/model/ckpt

```

Similarly as VBench, you can specify resolution, aspect ratio, sampling steps, flow, and llm-refine

```bash
bash eval/vbench_i2v/launch.sh /path/to/ckpt num_frames model_name_for_log res_value aspect_ratio_value steps_value flow_value llm_refine_value
# for example
# bash eval/vbench_i2v/launch.sh /mnt/jfs-hdd/sora/checkpoints/outputs/042-STDiT3-XL-2/epoch1-global_step16200_llm_refine/ema.pt 51 042-STDiT3-XL-2 240p 9:16 30 2 True
# if no flow control, use "None" instead
```

## VAE

Install the dependencies package following our [installation](../docs/installation.md)'s s sections of "Evaluation Dependencies". Then, run the following evaluation command:

```bash
# metric can any one or list of: ssim, psnr, lpips, flolpips
python eval/vae/eval_common_metric.py --batch_size 2 --real_video_dir path/to/original/videos --generated_video_dir path/to/generated/videos --device cuda --sample_fps 24 --crop_size 256 --resolution 256 --num_frames 17 --sample_rate 1 --metric ssim psnr lpips flolpips
```

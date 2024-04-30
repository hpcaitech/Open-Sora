# Evalution

## Human evaluation

To conduct human evaluation, we need to generate various samples. We provide many prompts in `assets/texts`, and defined some test setting covering different resolution, duration and aspect ratio in `eval/sample.sh`. To facilitate the usage of multiple GPUs, we split sampling tasks into several parts.

```bash
# image (1)
bash eval/sample.sh /path/to/ckpt -1
# video (2a 2b 2c ...)
bash eval/sample.sh /path/to/ckpt -2a
# launch 8 jobs at once (you must read the script to understand the details)
bash eval/launch.sh /path/to/ckpt
```

## VBench

[VBench](https://github.com/Vchitect/VBench) is a benchmark for short text to video generation. We provide a script for easily generating samples required by VBench.

```bash
# vbench tasks (4a 4b 4c ...)
bash eval/sample.sh /path/to/ckpt -4a
# launch 8 jobs at once (you must read the script to understand the details)
bash eval/launch.sh /path/to/ckpt
```

After generation, install the VBench package according to their [instructions](https://github.com/Vchitect/VBench?tab=readme-ov-file#hammer-installation). Then, run the following commands to evaluate the generated samples.

```bash
bash eval/vbench/vbench.sh /path/to/video_folder
```

## VBench-i2v

[VBench-i2v](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v) is a benchmark for short image to video generation (beta version).

TBD

## VAE

* Dependencies

- Install cupy: follow https://docs.cupy.dev/en/stable/install.html

- To use flolpips model, download from https://github.com/danier97/flolpips/blob/main/weights/v0.1/alex.pth and place it under: `eval/vae/flolpips/weights/v0.1/alex.pth`

``` bash
pip install decord
pip install pytorchvideo
pip install lpips
pip install scipy
# Also, if torchvision.transforms.augentation still use `functional_tensor` and cause error,change to use `_functional_tensor`, follow https://blog.csdn.net/lanxing147/article/details/136625264
```


* Commands: carefule to change the setting to training setting

```bash
# metric can either be: ssim, psnr, lpips, flolpips
python eval/vae/eval_common_metric.py --batch_size 2 --real_video_dir <path/to/original/videos> --generated_video_dir <path/to/generated/videos> --device cuda --sample_fps 24 --crop_size 256 --resolution 256 --num_frames 17 --sample_rate 1 --metric <metric>
```

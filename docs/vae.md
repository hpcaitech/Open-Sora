# VAE Report

As [Pixart-Sigma](https://arxiv.org/abs/2403.04692) finds that adapting to a new VAE is simple, we develop an additional temporal VAE.
Specifically, our VAE consists of a pipeline of a [spatial VAE](https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers) followed by a temporal VAE.
For the temporal VAE, we follow the implementation of [MAGVIT-v2](https://arxiv.org/abs/2310.05737), with the following modifications:

* We remove the architecture specific to the codebook.
* We do not use the discriminator, and use the VAE reconstruction loss, kl loss, and perceptual loss for training.
* In the last linear layer of the encoder, we scale down to a diagonal Gaussian Distribution of 4 channels, following our previously trained STDiT that takes in 4 channels input.
* Our decoder is symmetric to the encoder architecture.

## Training

We train the model in different stages.

We first train the temporal VAE only by freezing the spatial VAE for 380k steps on a single machine (8 GPUs).
We use an additional identity loss to make features from the 3D VAE similar to the features from the 2D VAE.
We train the VAE using 20% images and 80% videos with 17 frames.

```bash
torchrun --nnodes=1 --nproc_per_node=8 scripts/train_vae.py configs/vae/train/stage1.py --data-path YOUR_CSV_PATH
```

Next, we remove the identity loss and train the 3D VAE pipeline to reconstructe the 2D-compressed videos for 260k steps.

```bash
torchrun --nnodes=1 --nproc_per_node=8 scripts/train_vae.py configs/vae/train/stage2.py --data-path YOUR_CSV_PATH
```

Finally, we remove the reconstruction loss for the 2D-compressed videos and train the VAE pipeline to construct the 3D videos for 540k steps.
We train our VAE with a random number within 34 frames to make it more robust to different video lengths.
This stage is trained on 24 GPUs.

```bash
torchrun --nnodes=3 --nproc_per_node=8 scripts/train_vae.py configs/vae/train/stage3.py --data-path YOUR_CSV_PATH
```

Note that you need to adjust the `epochs` in the config file accordingly with respect to your own csv data size.

## Inference

To visually check the performance of the VAE, you may run the following inference.
It saves the original video to your specified video directory with `_ori` postfix (i.e. `"YOUR_VIDEO_DIR"_ori`), the reconstructed video from the full pipeline with the `_rec` postfix (i.e. `"YOUR_VIDEO_DIR"_rec`), and the reconstructed video from the 2D compression and decompression with the `_spatial` postfix (i.e. `"YOUR_VIDEO_DIR"_spatial`).

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/inference_vae.py configs/vae/inference/video.py --ckpt-path YOUR_VAE_CKPT_PATH --data-path YOUR_CSV_PATH --save-dir YOUR_VIDEO_DIR
```
## Evaluation

We can then calculate the scores of the VAE performances on metrics of SSIM, PSNR, LPIPS, and FLOLPIPS.

* SSIM: structural similarity index measure, the higher the better
* PSNR: peak-signal-to-noise ratio, the higher the better
* LPIPS:  learned perceptual image quality degradation, the lower the better
* [FloLPIPS](https://arxiv.org/pdf/2207.08119): LPIPS with video interpolation, the lower the better.

```bash
python eval/vae/eval_common_metric.py --batch_size 2 --real_video_dir YOUR_VIDEO_DIR_ori --generated_video_dir YOUR_VIDEO_DIR_rec --device cuda --sample_fps 24 --crop_size 256 --resolution 256 --num_frames 17 --sample_rate 1 --metric ssim psnr lpips flolpips
```

## Acknowledgement
We are grateful for the following work:
* [MAGVIT-v2](https://arxiv.org/abs/2310.05737): Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation
* [Taming Transformers](https://github.com/CompVis/taming-transformers): Taming Transformers for High-Resolution Image Synthesis
* [3D blur pooling](https://github.com/adobe/antialiased-cnns/pull/39/commits/3d6f02b6943c58b68c19c07bc26fad57492ff3bc)
* [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan)

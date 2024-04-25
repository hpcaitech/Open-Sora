# Open-Sora 1.0 Report

OpenAI's Sora is amazing at generating one minutes high quality videos. However, it reveals almost no information about its details. To make AI more "open", we are dedicated to build an open-source version of Sora. This report describes our first attempt to train a transformer-based video diffusion model.

## Efficiency in choosing the architecture

To lower the computational cost, we want to utilize existing VAE models. Sora uses spatial-temporal VAE to reduce the temporal dimensions. However, we found that there is no open-source high-quality spatial-temporal VAE model. [MAGVIT](https://github.com/google-research/magvit)'s 4x4x4 VAE is not open-sourced, while [VideoGPT](https://wilson1yan.github.io/videogpt/index.html)'s 2x4x4 VAE has a low quality in our experiments. Thus, we decided to use a 2D VAE (from [Stability-AI](https://huggingface.co/stabilityai/sd-vae-ft-mse-original)) in our first version.

The video training involves a large amount of tokens. Considering 24fps 1min videos, we have 1440 frames. With VAE downsampling 4x and patch size downsampling 2x, we have 1440x1024≈1.5M tokens. Full attention on 1.5M tokens leads to a huge computational cost. Thus, we use spatial-temporal attention to reduce the cost following [Latte](https://github.com/Vchitect/Latte).

As shown in the figure, we insert a temporal attention right after each spatial attention in STDiT (ST stands for spatial-temporal). This is similar to variant 3 in Latte's paper. However, we do not control a similar number of parameters for these variants. While Latte's paper claims their variant is better than variant 3, our experiments on 16x256x256 videos show that with same number of iterations, the performance ranks as: DiT (full) > STDiT (Sequential) > STDiT (Parallel) ≈ Latte. Thus, we choose STDiT (Sequential) out of efficiency. Speed benchmark is provided [here](/docs/acceleration.md#efficient-stdit).

![Architecture Comparison](/assets/readme/report_arch_comp.png)

To focus on video generation, we hope to train the model based on a powerful image generation model. [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha) is an efficiently trained high-quality image generation model with T5-conditioned DiT structure. We initialize our model with PixArt-α and initialize the projection layer of inserted temporal attention with zero. This initialization preserves model's ability of image generation at beginning, while Latte's architecture cannot. The inserted attention increases the number of parameter from 580M to 724M.

![Architecture](/assets/readme/report_arch.jpg)

Drawing from the success of PixArt-α and Stable Video Diffusion, we also adopt a progressive training strategy: 16x256x256 on 366K pretraining datasets, and then 16x256x256, 16x512x512, and 64x512x512 on 20K datasets. With scaled position embedding, this strategy greatly reduces the computational cost.

We also try to use a 3D patch embedder in DiT. However, with 2x downsampling on temporal dimension, the generated videos have a low quality. Thus, we leave the downsampling to temporal VAE in our next version. For now, we sample at every 3 frames with 16 frames training and every 2 frames with 64 frames training.

## Data is the key to high quality

We find that the number and quality of data have a great impact on the quality of generated videos, even larger than the model architecture and training strategy. At this time, we only prepared the first split (366K video clips) from [HD-VG-130M](https://github.com/daooshee/HD-VG-130M). The quality of these videos varies greatly, and the captions are not that accurate. Thus, we further collect 20k relatively high quality videos from [Pexels](https://www.pexels.com/), which provides free license videos. We label the video with LLaVA, an image captioning model, with three frames and a designed prompt. With designed prompt, LLaVA can generate good quality of captions.

![Caption](/assets/readme/report_caption.png)

As we lay more emphasis on the quality of data, we prepare to collect more data and build a video preprocessing pipeline in our next version.

## Training Details

With a limited training budgets, we made only a few exploration. We find learning rate 1e-4 is too large and scales down to 2e-5. When training with a large batch size, we find `fp16` less stable than `bf16` and may lead to generation failure. Thus, we switch to `bf16` for training on 64x512x512. For other hyper-parameters, we follow previous works.

## Loss curves

16x256x256 Pretraining Loss Curve

![16x256x256 Pretraining Loss Curve](/assets/readme/report_loss_curve_1.png)

16x256x256 HQ Training Loss Curve

![16x256x256 HQ Training Loss Curve](/assets/readme/report_loss_curve_2.png)

16x512x512 HQ Training Loss Curve

![16x512x512 HQ Training Loss Curve](/assets/readme/report_loss_curve_3.png)

> Core Contributor: Zangwei Zheng*, Xiangyu Peng*, Shenggui Li, Hongxing Liu, Yang You

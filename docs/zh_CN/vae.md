# VAE 技术报告

由于 [Pixart-Sigma](https://arxiv.org/abs/2403.04692) 论文中指出适应新的VAE很简单，因此我们开发了一个额外的时间VAE。
具体而言, 我们的VAE由一个[空间 VAE](https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers)和一个时间VA相接的形式组成.
对于时间VAE，我们遵循 [MAGVIT-v2](https://arxiv.org/abs/2310.05737)的实现, 并做了以下修改:

* 我们删除了码本特有的架构。
* 我们不使用鉴别​​器（discriminator），而是使用VAE重建损失、kl损失和感知损失进行训练。
* 在编码器的最后一个线性层中，我们缩小到 4 通道的对角高斯分布，遵循我们之前训练的接受 4 通道输入的 STDiT。
* 我们的解码器与编码器架构对称。

## 训练
我们分不同阶段训练模型。

我们首先通过在单台机器（8 个 GPU）上冻结空间 VAE 380k 步来训练时间 VAE。我们使用额外的身份损失使 3D VAE 的特征与 2D VAE 的特征相似。我们使用 20% 的图像和 80% 的视频（17 帧）来训练 VAE。

```bash
torchrun --nnodes=1 --nproc_per_node=8 scripts/train_vae.py configs/vae/train/stage1.py --data-path YOUR_CSV_PATH
```

接下来，我们移除身份损失并训练 3D VAE 管道以重建 260k 步的 2D 压缩视频。

```bash
torchrun --nnodes=1 --nproc_per_node=8 scripts/train_vae.py configs/vae/train/stage2.py --data-path YOUR_CSV_PATH
```

最后，我们移除了 2D 压缩视频的重建损失，并训练 VAE 管道以构建 540k 步的 3D 视频。我们在 34 帧内使用随机数训练 VAE，使其对不同长度的视频更具鲁棒性。此阶段在 24 个 GPU 上进行训练。

```bash
torchrun --nnodes=3 --nproc_per_node=8 scripts/train_vae.py configs/vae/train/stage3.py --data-path YOUR_CSV_PATH
```

请注意，您需要根据自己的 csv 数据大小相应地调整配置文件中的 `epochs` 。

## 推理

为了直观地检查 VAE 的性能，您可以运行以下推理。它使用 `_ori` 后缀（即 `"YOUR_VIDEO_DIR"_ori`）将原始视频保存到您指定的视频目录中，使用`_rec`后缀（即`"YOUR_VIDEO_DIR"_rec`）将来自完整管道的重建视频保存到指定的视频目录中，并使用 `_spatial`后缀（即`"YOUR_VIDEO_DIR"_spatial`）将来自 2D 压缩和解压缩的重建视频保存到指定的视频目录中。

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/inference_vae.py configs/vae/inference/video.py --ckpt-path YOUR_VAE_CKPT_PATH --data-path YOUR_CSV_PATH --save-dir YOUR_VIDEO_DIR
```
## 评估
然后，我们可以计算 VAE 在 SSIM、PSNR、LPIPS 和 FLOLPIPS 指标上的表现得分。

* SSIM: 结构相似性指数度量，越高越好
* PSNR: 峰值信噪比，越高越好
* LPIPS: 学习感知图像质量下降，越低越好
* [FloLPIPS](https://arxiv.org/pdf/2207.08119): 带有视频插值的LPIPS，越低越好。

```bash
python eval/vae/eval_common_metric.py --batch_size 2 --real_video_dir YOUR_VIDEO_DIR_ori --generated_video_dir YOUR_VIDEO_DIR_rec --device cuda --sample_fps 24 --crop_size 256 --resolution 256 --num_frames 17 --sample_rate 1 --metric ssim psnr lpips flolpips
```

## 致谢
我们非常感谢以下工作：
* [MAGVIT-v2](https://arxiv.org/abs/2310.05737): Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation
* [Taming Transformers](https://github.com/CompVis/taming-transformers): Taming Transformers for High-Resolution Image Synthesis
* [3D blur pooling](https://github.com/adobe/antialiased-cnns/pull/39/commits/3d6f02b6943c58b68c19c07bc26fad57492ff3bc)
* [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan)

# Open-Sora v1 技术报告

OpenAI的Sora在生成一分钟高质量视频方面非常出色。然而，它几乎没有透露任何关于其细节的信息。为了使人工智能更加“开放”，我们致力于构建一个开源版本的Sora。这份报告描述了我们第一次尝试训练一个基于Transformer的视频扩散模型。

## 选择高效的架构

为了降低计算成本，我们希望利用现有的VAE模型。Sora使用时空VAE来减少时间维度。然而，我们发现没有开源的高质量时空VAE模型。[MAGVIT](https://github.com/google-research/magvit)的4x4x4 VAE并未开源，而[VideoGPT](https://wilson1yan.github.io/videogpt/index.html)的2x4x4 VAE在我们的实验中质量较低。因此，我们决定在我们第一个版本中使用2D VAE（来自[Stability-AI](https://huggingface.co/stabilityai/sd-vae-ft-mse-original)）。

视频训练涉及大量的token。考虑到24fps的1分钟视频，我们有1440帧。通过VAE下采样4倍和patch大小下采样2倍，我们得到了1440x1024≈150万个token。在150万个token上进行全注意力计算将带来巨大的计算成本。因此，我们使用时空注意力来降低成本，这是遵循[Latte](https://github.com/Vchitect/Latte)的方法。

如图中所示，在STDiT（ST代表时空）中，我们在每个空间注意力之后立即插入一个时间注意力。这类似于Latte论文中的变种3。然而，我们并没有控制这些变体的相似数量的参数。虽然Latte的论文声称他们的变体比变种3更好，但我们在16x256x256视频上的实验表明，相同数量的迭代次数下，性能排名为：DiT（完整）> STDiT（顺序）> STDiT（并行）≈ Latte。因此，我们出于效率考虑选择了STDiT（顺序）。[这里](/docs/acceleration.md#efficient-stdit)提供了速度基准测试。


![Architecture Comparison](/assets/readme/report_arch_comp.png)

为了专注于视频生成，我们希望基于一个强大的图像生成模型来训练我们的模型。PixArt-α是一个经过高效训练的高质量图像生成模型，具有T5条件化的DiT结构。我们使用[PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha)初始化我们的模型，并将插入的时间注意力的投影层初始化为零。这种初始化在开始时保留了模型的图像生成能力，而Latte的架构则不能。插入的注意力将参数数量从5.8亿增加到7.24亿。

![Architecture](/assets/readme/report_arch.jpg)

借鉴PixArt-α和Stable Video Diffusion的成功，我们还采用了渐进式训练策略：在366K预训练数据集上进行16x256x256的训练，然后在20K数据集上进行16x256x256、16x512x512和64x512x512的训练。通过扩展位置嵌入，这一策略极大地降低了计算成本。

我们还尝试在DiT中使用3D patch嵌入器。然而，在时间维度上2倍下采样后，生成的视频质量较低。因此，我们将在下一版本中将下采样留给时间VAE。目前，我们在每3帧采样一次进行16帧训练，以及在每2帧采样一次进行64帧训练。


## 数据是训练高质量模型的核心

我们发现数据的数量和质量对生成视频的质量有很大的影响，甚至比模型架构和训练策略的影响还要大。目前，我们只从[HD-VG-130M](https://github.com/daooshee/HD-VG-130M)准备了第一批分割（366K个视频片段）。这些视频的质量参差不齐，而且字幕也不够准确。因此，我们进一步从提供免费许可视频的[Pexels](https://www.pexels.com/)收集了20k相对高质量的视频。我们使用LLaVA，一个图像字幕模型，通过三个帧和一个设计好的提示来标记视频。有了设计好的提示，LLaVA能够生成高质量的字幕。

![Caption](/assets/readme/report_caption.png)

由于我们更加注重数据质量，我们准备收集更多数据，并在下一版本中构建一个视频预处理流程。

## 训练细节

在有限的训练预算下，我们只进行了一些探索。我们发现学习率1e-4过大，因此将其降低到2e-5。在进行大批量训练时，我们发现`fp16`比`bf16`不太稳定，可能会导致生成失败。因此，我们在64x512x512的训练中切换到`bf16`。对于其他超参数，我们遵循了之前的研究工作。

## 损失曲线

16x256x256 预训练损失曲线

![16x256x256 Pretraining Loss Curve](/assets/readme/report_loss_curve_1.png)

16x256x256 高质量训练损失曲线

![16x256x256 HQ Training Loss Curve](/assets/readme/report_loss_curve_2.png)

16x512x512 高质量训练损失曲线

![16x512x512 HQ Training Loss Curve](/assets/readme/report_loss_curve_3.png)

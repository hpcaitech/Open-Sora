<p align="center">
    <img src="../../assets/readme/icon.png" width="250"/>
<p>

<div align="center">
    <a href="https://github.com/hpcaitech/Open-Sora/stargazers"><img src="https://img.shields.io/github/stars/hpcaitech/Open-Sora?style=social"></a>
    <a href="https://hpcaitech.github.io/Open-Sora/"><img src="https://img.shields.io/badge/Gallery-View-orange?logo=&amp"></a>
    <a href="https://discord.gg/shpbperhGs"><img src="https://img.shields.io/badge/Discord-join-blueviolet?logo=discord&amp"></a>
    <a href="https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-247ipg9fk-KRRYmUl~u2ll2637WRURVA"><img src="https://img.shields.io/badge/Slack-ColossalAI-blueviolet?logo=slack&amp"></a>
    <a href="https://twitter.com/yangyou1991/status/1769411544083996787?s=61&t=jT0Dsx2d-MS5vS9rNM5e5g"><img src="https://img.shields.io/badge/Twitter-Discuss-blue?logo=twitter&amp"></a>
    <a href="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png"><img src="https://img.shields.io/badge/微信-小助手加群-green?logo=wechat&amp"></a>
    <a href="https://hpc-ai.com/blog/open-sora-v1.0"><img src="https://img.shields.io/badge/Open_Sora-Blog-blue"></a>
</div>

## Open-Sora： 完全开源的高效复现类Sora视频生成方案
**Open-Sora**项目是一项致力于**高效**制作高质量视频，并使所有人都能使用其模型、工具和内容的计划。
通过采用**开源**原则，Open-Sora 不仅实现了先进视频生成技术的低成本普及，还提供了一个精简且用户友好的方案，简化了视频制作的复杂性。
通过 Open-Sora，我们希望更多开发者一起探索内容创作领域的创新、创造和包容。

[[English Document]](/README.md)

 <h4>Open-Sora 项目目前处在早期阶段，并将持续更新。</h4>

## 📰 资讯
> 由于文档需要进行翻译，最新资讯请看[英文文档](/README.md#-news)
* **[2024.04.25]** 🤗 我们在Hugging Face Spaces上发布了Open-Sora的[Gradio demo](https://huggingface.co/spaces/hpcai-tech/open-sora)。
* **[2024.04.25]** 🔥 我们发布了支持**2秒至15秒、144p至720p、任意宽高比**的文本到图像、文本到视频、图像到视频、视频到视频、无限时间生成的**Open-Sora 1.1**版本。此外，还发布了一个完整的视频处理流程。 [[checkpoints]]() [[report]](/docs/report_02.md)
* **[2024.03.18]** 🔥 我们发布了**Open-Sora 1.0**，这是一个完全开源的视频生成项目。
* Open-Sora 1.0 支持视频数据预处理、加速训练、推理等全套流程。
* 我们提供的[模型权重](#模型权重)只需 3 天的训练就能生成 2 秒的 512x512 视频。
* **[2024.03.04]** Open-Sora：开源Sora复现方案，成本降低46%，序列扩充至近百万。[[英文博客]](https://hpc-ai.com/blog/open-sora)

## 🎥 最新视频

| **2s 512×512**                                                                                                                                                                 | **2s 512×512**                                                                                                                                                              | **2s 512×512**                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| [<img src="/assets/readme/sample_0.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/de1963d3-b43b-4e68-a670-bb821ebb6f80)                                 | [<img src="/assets/readme/sample_1.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/13f8338f-3d42-4b71-8142-d234fbd746cc)                              | [<img src="/assets/readme/sample_2.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/fa6a65a6-e32a-4d64-9a9e-eabb0ebb8c16)    |
| A serene night scene in a forested area. [...] The video is a time-lapse, capturing the transition from day to night, with the lake and forest serving as a constant backdrop. | A soaring drone footage captures the majestic beauty of a coastal cliff, [...] The water gently laps at the rock base and the greenery that clings to the top of the cliff. | The majestic beauty of a waterfall cascading down a cliff into a serene lake. [...] The camera angle provides a bird's eye view of the waterfall. |
| [<img src="/assets/readme/sample_3.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/64232f84-1b36-4750-a6c0-3e610fa9aa94) | [<img src="/assets/readme/sample_4.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/983a1965-a374-41a7-a76b-c07941a6c1e9) | [<img src="/assets/readme/sample_5.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/ec10c879-9767-4c31-865f-2e8d6cf11e65) |
| A bustling city street at night, filled with the glow of car headlights and the ambient light of streetlights. [...]                                                           | The vibrant beauty of a sunflower field. The sunflowers are arranged in neat rows, creating a sense of order and symmetry. [...]                                            | A serene underwater scene featuring a sea turtle swimming through a coral reef. The turtle, with its greenish-brown shell [...]                   |

视频经过降采样处理为`.gif`格式，以便显示。点击查看原始视频。为便于显示，文字经过修剪，全文请参见 [此处](/assets/texts/t2v_samples.txt)。在我们的[图片库](https://hpcaitech.github.io/Open-Sora/)中查看更多样本。

## 🔆 新功能
> 由于文档需要进行翻译，最新资讯请看[英文文档](/README.md#-new-featuresupdates)
* 📍Open-Sora-v1 已发布。[这里](#模型权重)提供了模型权重。只需 400K 视频片段和在单卡 H800 上训200天（类比Stable Video Diffusion 的 152M 样本），我们就能生成 2 秒的 512×512 视频。
* ✅ 从图像扩散模型到视频扩散模型的三阶段训练。我们提供每个阶段的权重。
* ✅ 支持训练加速，包括Transformer加速、更快的 T5 和 VAE 以及序列并行。在对 64x512x512 视频进行训练时，Open-Sora 可将训练速度提高**55%**。详细信息请参见[训练加速](acceleration.md)。
* 🔧 我们提供用于数据预处理的视频切割和字幕工具。有关说明请点击[此处](tools/data/README.md)，我们的数据收集计划请点击 [数据集](datasets.md)。
* ✅ 我们发现来自[VideoGPT](https://wilson1yan.github.io/videogpt/index.html)的 VQ-VAE 质量较低，因此采用了来自[Stability-AI](https://huggingface.co/stabilityai/sd-vae-ft-mse-original) 的高质量 VAE。我们还发现使用添加了时间维度的采样会导致生成质量降低。更多讨论，请参阅我们的 **[报告](docs/report_v1.md)**。
* ✅ 我们研究了不同的架构，包括 DiT、Latte 和我们提出的 **STDiT**。我们的STDiT在质量和速度之间实现了更好的权衡。更多讨论，请参阅我们的 **[报告](report_v1.md)**。
* ✅ 支持剪辑和 T5 文本调节。
* ✅ 通过将图像视为单帧视频，我们的项目支持在图像和视频（如 ImageNet 和 UCF101）上训练 DiT。更多说明请参见 [指令解析](command.md)。
* ✅ 利用[DiT](https://github.com/facebookresearch/DiT)、[Latte](https://github.com/Vchitect/Latte) 和 [PixArt](https://pixart-alpha.github.io/) 的官方权重支持推理。

<details>
<summary>查看更多</summary>

* ✅ 重构代码库。请参阅[结构](structure.md)，了解项目结构以及如何使用配置文件。

</details>

### 下一步计划【按优先级排序】

* [ ] 训练视频-VAE并让模型适应新的VAE **[项目进行中]**
* [ ] 缩放模型参数和数据集大小 **[项目进行中]**
* [ ] 纳入更好的时间表，例如 SD3 中的修正流程。 **[项目进行中]**

<details>
<summary>查看更多</summary>

* [x] 评估流程。
* [x] 完成数据处理流程（包括密集光流、美学评分、文本图像相似性、重复数据删除等）。更多信息请参见[数据集](datasets.md)
* [x] 支持图像和视频调节。
* [x] 支持可变长宽比、分辨率和持续时间。

</details>

## 目录

* [安装](#安装)
* [模型权重](#模型权重)
* [推理](#推理)
* [数据处理](#数据处理)
* [训练](#训练)
* [评估](#评估)
* [贡献](#贡献)
* [声明](#声明)
* [引用](#引用)

## 安装

### 从源码安装
```bash
# create a virtual env
conda create -n opensora python=3.10

# install torch
# the command below is for CUDA 12.1, choose install commands from
# https://pytorch.org/get-started/locally/ based on your own CUDA version
pip3 install torch torchvision

# install flash attention (optional)
pip install packaging ninja
pip install flash-attn --no-build-isolation

# install apex (optional)
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

# install xformers
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121

# install this project
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora
pip install -v .
```

### 使用Docker镜像

运行如下指令使用提供的Dockerfile构建镜像：

```bash
docker build -t opensora ./docker
```

运行以下命令以启动交互模式下的 Docker 容器：

```bash
docker run -ti --gpus all -v {MOUNT_DIR}:/data opensora
```

安装完成后，建议阅读[结构](structure.md)，了解项目结构以及如何使用配置文件。

## 模型权重

| 分辨率  | 数据   | 迭代次数 | 批量大小 | GPU 天数 (H800) | 网址       |
| ---------- | ------ | ----------- | ---------- | --------------- | ---------- |
| 16×256×256 | 366K   | 80k         | 8×64       | 117             | [:link:]() |
| 16×256×256 | 20K HQ | 24k         | 8×64       | 45              | [:link:]() |
| 16×512×512 | 20K HQ | 20k         | 2×64       | 35              | [:link:]() |

我们模型的权重部分由[PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha) 初始化。参数数量为 724M。有关训练的更多信息，请参阅我们的 **[报告](report_v1.md)**。有关数据集的更多信息，请参阅[数据](datasets.md)。HQ 表示高质量。
:warning: **局限性**：我们的模型是在有限的预算内训练出来的。质量和文本对齐度相对较差。特别是在生成人类时，模型表现很差，无法遵循详细的指令。我们正在努力改进质量和文本对齐。

## 推理

要使用我们提供的权重进行推理，首先要将[T5](https://huggingface.co/DeepFloyd/t5-v1_1-xxl/tree/main)权重下载到pretrained_models/t5_ckpts/t5-v1_1-xxl 中。然后下载模型权重。运行以下命令生成样本。请参阅[此处](structure.md#推理配置演示)自定义配置。

```bash
# Sample 16x512x512 (20s/sample, 100 time steps, 24 GB memory)
torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/opensora/inference/16x512x512.py --ckpt-path OpenSora-v1-HQ-16x512x512.pth --prompt-path ./assets/texts/t2v_samples.txt

# Sample 16x256x256 (5s/sample, 100 time steps, 22 GB memory)
torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/opensora/inference/16x256x256.py --ckpt-path OpenSora-v1-HQ-16x256x256.pth --prompt-path ./assets/texts/t2v_samples.txt

# Sample 64x512x512 (40s/sample, 100 time steps)
torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/opensora/inference/64x512x512.py --ckpt-path ./path/to/your/ckpt.pth --prompt-path ./assets/texts/t2v_samples.txt

# Sample 64x512x512 with sequence parallelism (30s/sample, 100 time steps)
# sequence parallelism is enabled automatically when nproc_per_node is larger than 1
torchrun --standalone --nproc_per_node 2 scripts/inference.py configs/opensora/inference/64x512x512.py --ckpt-path ./path/to/your/ckpt.pth --prompt-path ./assets/texts/t2v_samples.txt

```

我们在 H800 GPU 上进行了速度测试。如需使用其他模型进行推理，请参阅[此处](commands.md)获取更多说明。减小`vae.micro_batch_size`来降低显存使用（但取样速度会略微减慢）。

## 数据处理

高质量数据是高质量模型的关键。[这里](datasets.md)有我们使用过的数据集和数据收集计划。我们提供处理视频数据的工具。目前，我们的数据处理流程包括以下步骤：

1. 下载数据集。[[文件](/tools/datasets/README.md)]
2. 将视频分割成片段。 [[文件](/tools/scene_cut/README.md)]
3. 生成视频字幕。 [[文件](/tools/caption/README.md)]

## 训练

### Open-Sora 1.0 训练
<details>
<summary>查看更多</summary>

要启动训练，首先要将[T5](https://huggingface.co/DeepFloyd/t5-v1_1-xxl/tree/main)权重下载到pretrained_models/t5_ckpts/t5-v1_1-xxl 中。然后运行以下命令在单个节点上启动训练。

```bash
# 1 GPU, 16x256x256
torchrun --nnodes=1 --nproc_per_node=1 scripts/train.py configs/opensora/train/16x256x512.py --data-path YOUR_CSV_PATH
# 8 GPUs, 64x512x512
torchrun --nnodes=1 --nproc_per_node=8 scripts/train.py configs/opensora/train/64x512x512.py --data-path YOUR_CSV_PATH --ckpt-path YOUR_PRETRAINED_CKPT
```

要在多个节点上启动训练，请根据[ColossalAI](https://colossalai.org/docs/basics/launch_colossalai/#launch-with-colossal-ai-cli) 准备一个主机文件，并运行以下命令。

```bash
colossalai run --nproc_per_node 8 --hostfile hostfile scripts/train.py configs/opensora/train/64x512x512.py --data-path YOUR_CSV_PATH --ckpt-path YOUR_PRETRAINED_CKPT
```

有关其他模型的训练和高级使用方法，请参阅[此处](commands.md)获取更多说明。

</details>

## 评估

点击[这里](https://github.com/hpcaitech/Open-Sora/blob/main/eval/README.md)查看评估

## 贡献

本中文翻译还有许多不足，如果您希望为该项目做出贡献，可以参考 [贡献指南](/CONTRIBUTING.md).

目前需要翻译或更新的文件：
* [ ] 更新[资讯](#-资讯)
* [ ] 更新[最新视频](#-最新视频)
* [ ] 更新[新功能](#-新功能)。
* [ ] 翻译[评估](https://github.com/hpcaitech/Open-Sora/blob/main/eval/README.md)文件
* [ ] 更新Open-Sora 1.1[训练](#训练)
## 声明

* [ColossalAI](https://github.com/hpcaitech/ColossalAI): A powerful large model parallel acceleration and optimization
* [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers.
* [OpenDiT](https://github.com/NUS-HPC-AI-Lab/OpenDiT): An acceleration for DiT training. We adopt valuable acceleration strategies for training progress from OpenDiT.
* [PixArt](https://github.com/PixArt-alpha/PixArt-alpha): An open-source DiT-based text-to-image model.
* [Latte](https://github.com/Vchitect/Latte): An attempt to efficiently train DiT for video.
* [StabilityAI VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse-original): A powerful image VAE model.
* [CLIP](https://github.com/openai/CLIP): A powerful text-image embedding model.
* [T5](https://github.com/google-research/text-to-text-transfer-transformer): A powerful text encoder.
* [LLaVA](https://github.com/haotian-liu/LLaVA): A powerful image captioning model based on [Yi-34B](https://huggingface.co/01-ai/Yi-34B).

我们对他们的出色工作和对开源的慷慨贡献表示感谢。

## 引用

```bibtex
@software{opensora,
  author = {Zangwei Zheng and Xiangyu Peng and Yang You},
  title = {Open-Sora: Democratizing Efficient Video Production for All},
  month = {March},
  year = {2024},
  url = {https://github.com/hpcaitech/Open-Sora}
}
```

[Zangwei Zheng](https://github.com/zhengzangw) and [Xiangyu Peng](https://github.com/xyupeng) equally contributed to this work during their internship at [HPC-AI Tech](https://hpc-ai.com/).

## Star 走势

[![Star History Chart](https://api.star-history.com/svg?repos=hpcaitech/Open-Sora&type=Date)](https://star-history.com/#hpcaitech/Open-Sora&Date)

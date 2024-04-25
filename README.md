<p align="center">
    <img src="./assets/readme/icon.png" width="250"/>
</p>
<div align="center">
    <a href="https://github.com/hpcaitech/Open-Sora/stargazers"><img src="https://img.shields.io/github/stars/hpcaitech/Open-Sora?style=social"></a>
    <a href="https://hpcaitech.github.io/Open-Sora/"><img src="https://img.shields.io/badge/Gallery-View-orange?logo=&amp"></a>
    <a href="https://discord.gg/kZakZzrSUT"><img src="https://img.shields.io/badge/Discord-join-blueviolet?logo=discord&amp"></a>
    <a href="https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-247ipg9fk-KRRYmUl~u2ll2637WRURVA"><img src="https://img.shields.io/badge/Slack-ColossalAI-blueviolet?logo=slack&amp"></a>
    <a href="https://twitter.com/yangyou1991/status/1769411544083996787?s=61&t=jT0Dsx2d-MS5vS9rNM5e5g"><img src="https://img.shields.io/badge/Twitter-Discuss-blue?logo=twitter&amp"></a>
    <a href="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png"><img src="https://img.shields.io/badge/微信-小助手加群-green?logo=wechat&amp"></a>
    <a href="https://hpc-ai.com/blog/open-sora-v1.0"><img src="https://img.shields.io/badge/Open_Sora-Blog-blue"></a>
    <a href="https://huggingface.co/spaces/hpcai-tech/open-sora"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Gradio Demo-blue"></a>
</div>

## Open-Sora: Democratizing Efficient Video Production for All

We present **Open-Sora**, an initiative dedicated to **efficiently** produce high-quality video and make the model,
tools and contents accessible to all. By embracing **open-source** principles,
Open-Sora not only democratizes access to advanced video generation techniques, but also offers a
streamlined and user-friendly platform that simplifies the complexities of video production.
With Open-Sora, we aim to inspire innovation, creativity, and inclusivity in the realm of content creation.

[[中文文档]](/docs/zh_CN/README.md) [[潞晨云部署视频教程]](https://www.bilibili.com/video/BV141421R7Ag)

<h4>Open-Sora is still at an early stage and under active development.</h4>

## 📰 News

* **[2024.04.25]** 🤗 We released the [Gradio demo for Open-Sora](https://huggingface.co/spaces/hpcai-tech/open-sora) on Hugging Face Spaces.
* **[2024.04.25]** 🔥 We released **Open-Sora 1.1**, which supports **2s~15s, 144p to 720p, any aspect ratio** text-to-image, **text-to-video, image-to-video, video-to-video, infinite time** generation. In addition, a full video processing pipeline is released. [[checkpoints]]() [[report]](/docs/report_02.md)
* **[2024.03.18]** We released **Open-Sora 1.0**, a fully open-source project for video generation.
  Open-Sora 1.0 supports a full pipeline of video data preprocessing, training with
  <a href="https://github.com/hpcaitech/ColossalAI"><img src="assets/readme/colossal_ai.png" width="8%" ></a>
  acceleration,
  inference, and more. Our model can produce 2s 512x512 videos with only 3 days training. [[checkpoints]](#open-sora-10-model-weights)
  [[blog]](https://hpc-ai.com/blog/open-sora-v1.0) [[report]](docs/report_01.md)
* **[2024.03.04]** Open-Sora provides training with 46% cost reduction.
  [[blog]](https://hpc-ai.com/blog/open-sora)

## 🎥 Latest Demo

🔥 You can experinece Open-Sora on our [🤗 Gradio application on Hugging Face](https://huggingface.co/spaces/hpcai-tech/open-sora). More samples are available in our [Gallery](https://hpcaitech.github.io/Open-Sora/).

| **2s 240×426**                                                                                                                                              | **2s 240×426**                                                                                                                                             |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [<img src="assets/demo/sample_16x240x426_9.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/c31ebc52-de39-4a4e-9b1e-9211d45e05b2) | [<img src="assets/demo/sora_16x240x426_26.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/c31ebc52-de39-4a4e-9b1e-9211d45e05b2) |
| [<img src="assets/demo/sora_16x240x426_27.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/f7ce4aaa-528f-40a8-be7a-72e61eaacbbd)  | [<img src="assets/demo/sora_16x240x426_40.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/5d58d71e-1fda-4d90-9ad3-5f2f7b75c6a9) |

| **2s 426×240**                                                                                                                                             | **4s 480×854**                                                                                                                                              |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [<img src="assets/demo/sora_16x426x240_24.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/34ecb4a0-4eef-4286-ad4c-8e3a87e5a9fd) | [<img src="assets/demo/sample_32x480x854_9.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/c1619333-25d7-42ba-a91c-18dbc1870b18) |

| **16s 320×320**                                                                                                                                        | **16s 224×448**                                                                                                                                        | **2s 426×240**                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [<img src="assets/demo/sample_16s_320x320.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/3cab536e-9b43-4b33-8da8-a0f9cf842ff2) | [<img src="assets/demo/sample_16s_224x448.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/9fb0b9e0-c6f4-4935-b29e-4cac10b373c4) | [<img src="assets/demo/sora_16x426x240_3.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/3e892ad2-9543-4049-b005-643a4c1bf3bf) |
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes

<details>
<summary>OpenSora 1.0 Demo</summary>

| **2s 512×512**                                                                                                                                                                 | **2s 512×512**                                                                                                                                                              | **2s 512×512**                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| [<img src="assets/readme/sample_0.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/de1963d3-b43b-4e68-a670-bb821ebb6f80)                                 | [<img src="assets/readme/sample_1.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/13f8338f-3d42-4b71-8142-d234fbd746cc)                              | [<img src="assets/readme/sample_2.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/fa6a65a6-e32a-4d64-9a9e-eabb0ebb8c16)    |
| A serene night scene in a forested area. [...] The video is a time-lapse, capturing the transition from day to night, with the lake and forest serving as a constant backdrop. | A soaring drone footage captures the majestic beauty of a coastal cliff, [...] The water gently laps at the rock base and the greenery that clings to the top of the cliff. | The majestic beauty of a waterfall cascading down a cliff into a serene lake. [...] The camera angle provides a bird's eye view of the waterfall. |
| [<img src="assets/readme/sample_3.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/64232f84-1b36-4750-a6c0-3e610fa9aa94)                                 | [<img src="assets/readme/sample_4.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/983a1965-a374-41a7-a76b-c07941a6c1e9)                              | [<img src="assets/readme/sample_5.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/ec10c879-9767-4c31-865f-2e8d6cf11e65)    |
| A bustling city street at night, filled with the glow of car headlights and the ambient light of streetlights. [...]                                                           | The vibrant beauty of a sunflower field. The sunflowers are arranged in neat rows, creating a sense of order and symmetry. [...]                                            | A serene underwater scene featuring a sea turtle swimming through a coral reef. The turtle, with its greenish-brown shell [...]                   |

Videos are downsampled to `.gif` for display. Click for original videos. Prompts are trimmed for display,
see [here](/assets/texts/t2v_samples.txt) for full prompts.

</details>

## 🔆 New Features/Updates

* 📍 **Open-Sora 1.1** released. Model weights are available [here](). It is trained on **0s~15s, 144p to 720p, various aspect ratios** videos. See our **[report 1.1](docs/report_02.md)** for more discussions.
* 🔧 **Data processing pipeline v1.1** is released. An automatic [processing pipeline](#data-processing) from raw videos to (text, video clip) pairs is provided, including scene cutting $\rightarrow$ filtering(aesthetic, optical flow, OCR, etc.) $\rightarrow$ captioning $\rightarrow$ managing. With this tool, you can easily build your video dataset.
* ✅ Modified ST-DiT architecture includes rope positional encoding, qk norm, longer text length, etc.
* ✅ Support training with any resolution, aspect ratio, and duration (including images).
* ✅ Support image and video conditioning and video editing, and thus support animating images, connecting videos, etc.
* 📍 **Open-Sora 1.0** released. Model weights are available [here](#model-weights). With only 400K video clips and 200 H800
  days (compared with 152M samples in Stable Video Diffusion), we are able to generate 2s 512×512 videos. See our **[report 1.0](docs/report_01.md)** for more discussions.
* ✅ Three-stage training from an image diffusion model to a video diffusion model. We provide the weights for each
  stage.
* ✅ Support training acceleration including accelerated transformer, faster T5 and VAE, and sequence parallelism.
  Open-Sora improve **55%** training speed when training on 64x512x512 videos. Details locates
  at [acceleration.md](docs/acceleration.md).
* 🔧 **Data preprocessing pipeline v1.0**,
  including [downloading](/tools/datasets/README.md), [video cutting](/tools/scenedetect/README.md),
  and [captioning](/tools/caption/README.md) tools. Our data collection plan can be found
  at [datasets.md](docs/datasets.md).

<details>
<summary>View more</summary>

* ✅ We find VQ-VAE from [VideoGPT](https://wilson1yan.github.io/videogpt/index.html) has a low quality and thus adopt a
  better VAE from [Stability-AI](https://huggingface.co/stabilityai/sd-vae-ft-mse-original). We also find patching in
  the time dimension deteriorates the quality. See our **[report](docs/report_01.md)** for more discussions.
* ✅ We investigate different architectures including DiT, Latte, and our proposed STDiT. Our **STDiT** achieves a better
  trade-off between quality and speed. See our **[report](docs/report_01.md)** for more discussions.
* ✅ Support clip and T5 text conditioning.
* ✅ By viewing images as one-frame videos, our project supports training DiT on both images and videos (e.g., ImageNet &
  UCF101). See [commands.md](docs/commands.md) for more instructions.
* ✅ Support inference with official weights
  from [DiT](https://github.com/facebookresearch/DiT), [Latte](https://github.com/Vchitect/Latte),
  and [PixArt](https://pixart-alpha.github.io/).
* ✅ Refactor the codebase. See [structure.md](docs/structure.md) to learn the project structure and how to use the
  config files.

</details>

### TODO list sorted by priority

* [ ] Training Video-VAE and adapt our model to new VAE. **[WIP]**
* [ ] Scaling model parameters and dataset size. **[WIP]**
* [ ] Incoporate a better scheduler, e.g., rectified flow in SD3. **[WIP]**

<details>
<summary>View more</summary>

* [x] Evaluation pipeline.
* [x] Complete the data processing pipeline (including dense optical flow, aesthetics scores, text-image similarity, etc.).
* [x] Support image and video conditioning.
* [x] Support variable aspect ratios, resolutions, durations.

</details>

## Contents

* [Installation](#installation)
* [Model Weights](#model-weights)
* [Inference](#inference)
* [Data Processing](#data-processing)
* [Training](#training)
* [Evaluation](#evaluation)
* [Contribution](#contribution)
* [Acknowledgement](#acknowledgement)

Other useful documents and links are listed below.

* Report: [report 1.1](docs/report_02.md), [report 1.0](docs/report_01.md), [acceleration.md](docs/acceleration.md)
* Repo structure: [structure.md](docs/structure.md)
* Config file explanation: [config.md](docs/config.md)
* Useful commands: [commands.md](docs/commands.md)
* Data processing pipeline and dataset: [datasets.md](docs/datasets.md)
* Each data processing tool's README: [dataset conventions and management](/tools/datasets/README.md), [scene cutting](/tools/scene_cut/README.md), [scoring](/tools/scoring/README.md), [caption](/tools/caption/README.md)
* Evaluation: [eval](/eval/README.md)
* Gallery: [gallery](https://hpcaitech.github.io/Open-Sora/)

## Installation

```bash
# create a virtual env
conda create -n opensora python=3.10
# activate virtual environment
conda activate opensora

# install torch
# the command below is for CUDA 12.1, choose install commands from
# https://pytorch.org/get-started/locally/ based on your own CUDA version
pip install torch torchvision

# install flash attention (optional)
# set enable_flashattn=False in config to avoid using flash attention
pip install packaging ninja
pip install flash-attn --no-build-isolation

# install apex (optional)
# set enable_layernorm_kernel=False in config to avoid using apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

# install xformers
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121

# install this project
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora
pip install -v .
```

## Model Weights

### Open-Sora 1.1 Model Weights

| Resolution         | Model Size | Data                       | #iterations | Batch Size                                        | URL                                                                  |
| ------------------ | ---------- | -------------------------- | ----------- | ------------------------------------------------- | -------------------------------------------------------------------- |
| mainly 144p & 240p | 700M       | 10M videos + 2M images     | 100k        | [dynamic](/configs/opensora-v1-1/train/stage2.py) | [:link:](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage2) |
| 144p to 720p       | 700M       | 500K HQ videos + 1M images | 4k          | [dynamic](/configs/opensora-v1-1/train/stage3.py) | [:link:](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage3) |

See our **[report 1.1](docs/report_02.md)** for more infomation.

:warning: **LIMITATION**: This version contains known issues which we are going to fix in the next version (as we save computation resource for the next release). In addition, the video generation may fail for long duration, and high resolution will have noisy results due to this problem.

### Open-Sora 1.0 Model Weights

<details>
<summary>View more</summary>

| Resolution | Model Size | Data   | #iterations | Batch Size | GPU days (H800) | URL                                                                                           |
| ---------- | ---------- | ------ | ----------- | ---------- | --------------- |
| 16×512×512 | 700M       | 20K HQ | 20k         | 2×64       | 35              | [:link:](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x512x512.pth) |
| 16×256×256 | 700M       | 20K HQ | 24k         | 8×64       | 45              | [:link:](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x256x256.pth) |
| 16×256×256 | 700M       | 366K   | 80k         | 8×64       | 117             | [:link:](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-16x256x256.pth)    |

Training orders: 16x256x256 $\rightarrow$ 16x256x256 HQ $\rightarrow$ 16x512x512 HQ.

Our model's weight is partially initialized from [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha). The number of
parameters is 724M. More information about training can be found in our **[report](/docs/report_01.md)**. More about
the dataset can be found in [datasets.md](/docs/datasets.md). HQ means high quality.

:warning: **LIMITATION**: Our model is trained on a limited budget. The quality and text alignment is relatively poor.
The model performs badly, especially on generating human beings and cannot follow detailed instructions. We are working
on improving the quality and text alignment.

</details>

## Inference

### Gradio Demo

🔥 You can experinece Open-Sora on our [🤗 Gradio application](https://huggingface.co/spaces/hpcai-tech/open-sora) on Hugging Face online.

If you want to deploy gradio locally, we have also provided a [Gradio application](./gradio) in this repository, you can use the following the command to start an interactive web application to experience video generation with Open-Sora.

```bash
pip install gradio spaces
python gradio/app.py
```

This will launch a Gradio application on your localhost. If you want to know more about the Gradio applicaiton, you can refer to the [README file](./gradio/README.md).

### Open-Sora 1.1 Command Line Inference

Since Open-Sora 1.1 supports inference with dynamic input size, you can pass the input size as an argument.

```bash
# text to video
python scripts/inference.py configs/opensora-v1-1/inference/sample.py \
    --ckpt-path CKPT_PATH --prompt "A beautiful sunset over the city" --num-frames 32 --image-size 480 854
```

See [here](docs/commands.md#inference-with-open-sora-11) for more instructions including text-to-image, image-to-video, video-to-video, and infinite time generation.

### Open-Sora 1.0 Command Line Inference

<details>
<summary>View more</summary>

We have also provided an offline inference script. Run the following commands to generate samples, the required model weights will be automatically downloaded. To change sampling prompts, modify the txt file passed to `--prompt-path`. See [here](docs/structure.md#inference-config-demos) to customize the configuration.

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

The speed is tested on H800 GPUs. For inference with other models, see [here](docs/commands.md) for more instructions.
To lower the memory usage, set a smaller `vae.micro_batch_size` in the config (slightly lower sampling speed).

</details>

## Data Processing

High-quality data is crucial for training good generation models.
To this end, we establish a complete pipeline for data processing, which could seamlessly convert raw videos to high-quality video-text pairs.
The pipeline is shown below. For detailed information, please refer to [data processing](docs/data_processing.md).
Also check out the [datasets](docs/datasets.md) we use.

![Data Processing Pipeline](assets/readme/report_data_pipeline.png)

## Training

### Open-Sora 1.1 Training

Once you prepare the data in a `csv` file, run the following commands to launch training on a single node.

```bash
# one node
torchrun --standalone --nproc_per_node 8 scripts/train.py \
    configs/opensora-v1-1/train/stage1.py --data-path YOUR_CSV_PATH --ckpt-path YOUR_PRETRAINED_CKPT
# multiple nodes
colossalai run --nproc_per_node 8 --hostfile hostfile scripts/train.py \
    configs/opensora-v1-1/train/stage1.py --data-path YOUR_CSV_PATH --ckpt-path YOUR_PRETRAINED_CKPT
```

### Open-Sora 1.0 Training

<details>
<summary>View more</summary>

Once you prepare the data in a `csv` file, run the following commands to launch training on a single node.

```bash
# 1 GPU, 16x256x256
torchrun --nnodes=1 --nproc_per_node=1 scripts/train.py configs/opensora/train/16x256x256.py --data-path YOUR_CSV_PATH
# 8 GPUs, 64x512x512
torchrun --nnodes=1 --nproc_per_node=8 scripts/train.py configs/opensora/train/64x512x512.py --data-path YOUR_CSV_PATH --ckpt-path YOUR_PRETRAINED_CKPT
```

To launch training on multiple nodes, prepare a hostfile according
to [ColossalAI](https://colossalai.org/docs/basics/launch_colossalai/#launch-with-colossal-ai-cli), and run the
following commands.

```bash
colossalai run --nproc_per_node 8 --hostfile hostfile scripts/train.py configs/opensora/train/64x512x512.py --data-path YOUR_CSV_PATH --ckpt-path YOUR_PRETRAINED_CKPT
```

For training other models and advanced usage, see [here](docs/commands.md) for more instructions.

</details>

## Evaluation

See [here](eval/README.md) for more instructions.

## Contribution

Thanks goes to these wonderful contributors ([emoji key](https://allcontributors.org/docs/en/emoji-key)
following [all-contributors](https://github.com/all-contributors/all-contributors) specification):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zhengzangw"><img src="https://avatars.githubusercontent.com/zhengzangw?v=4?s=100" width="100px;" alt="zhengzangw"/><br /><sub><b>zhengzangw</b></sub></a><br /><a href="https://github.com/hpcaitech/Open-Sora/commits?author=zhengzangw" title="Code">💻</a> <a href="https://github.com/hpcaitech/Open-Sora/commits?author=zhengzangw" title="Documentation">📖</a> <a href="#ideas-zhengzangw" title="Ideas, Planning, & Feedback">🤔</a> <a href="#video-zhengzangw" title="Videos">📹</a> <a href="#maintenance-zhengzangw" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ver217"><img src="https://avatars.githubusercontent.com/ver217?v=4?s=100" width="100px;" alt="ver217"/><br /><sub><b>ver217</b></sub></a><br /><a href="https://github.com/hpcaitech/Open-Sora/commits?author=ver217" title="Code">💻</a> <a href="#ideas-ver217" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/hpcaitech/Open-Sora/commits?author=ver217" title="Documentation">📖</a> <a href="#bug-ver217" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/FrankLeeeee"><img src="https://avatars.githubusercontent.com/FrankLeeeee?v=4?s=100" width="100px;" alt="FrankLeeeee"/><br /><sub><b>FrankLeeeee</b></sub></a><br /><a href="https://github.com/hpcaitech/Open-Sora/commits?author=FrankLeeeee" title="Code">💻</a> <a href="#infra-FrankLeeeee" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#tool-FrankLeeeee" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/xyupeng"><img src="https://avatars.githubusercontent.com/xyupeng?v=4?s=100" width="100px;" alt="xyupeng"/><br /><sub><b>xyupeng</b></sub></a><br /><a href="https://github.com/hpcaitech/Open-Sora/commits?author=xyupeng" title="Code">💻</a> <a href="#doc-xyupeng" title="Documentation">📖</a> <a href="#design-xyupeng" title="Design">🎨</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Yanjia0"><img src="https://avatars.githubusercontent.com/Yanjia0?v=4?s=100" width="100px;" alt="Yanjia0"/><br /><sub><b>Yanjia0</b></sub></a><br /><a href="#doc-Yanjia0" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/binmakeswell"><img src="https://avatars.githubusercontent.com/binmakeswell?v=4?s=100" width="100px;" alt="binmakeswell"/><br /><sub><b>binmakeswell</b></sub></a><br /><a href="#doc-binmakeswell" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/eltociear"><img src="https://avatars.githubusercontent.com/eltociear?v=4?s=100" width="100px;" alt="eltociear"/><br /><sub><b>eltociear</b></sub></a><br /><a href="#doc-eltociear" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ganeshkrishnan1"><img src="https://avatars.githubusercontent.com/ganeshkrishnan1?v=4?s=100" width="100px;" alt="ganeshkrishnan1"/><br /><sub><b>ganeshkrishnan1</b></sub></a><br /><a href="#doc-ganeshkrishnan1" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/fastalgo"><img src="https://avatars.githubusercontent.com/fastalgo?v=4?s=100" width="100px;" alt="fastalgo"/><br /><sub><b>fastalgo</b></sub></a><br /><a href="#doc-fastalgo" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/powerzbt"><img src="https://avatars.githubusercontent.com/powerzbt?v=4?s=100" width="100px;" alt="powerzbt"/><br /><sub><b>powerzbt</b></sub></a><br /><a href="#doc-powerzbt" title="Documentation">📖</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

If you wish to contribute to this project, you can refer to the [Contribution Guideline](./CONTRIBUTING.md).

[Zangwei Zheng](https://github.com/zhengzangw) and [Xiangyu Peng](https://github.com/xyupeng) equally contributed to
this work during their internship at [HPC-AI Tech](https://hpc-ai.com/).

## Acknowledgement

* [ColossalAI](https://github.com/hpcaitech/ColossalAI): A powerful large model parallel acceleration and optimization
  system.
* [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers.
* [OpenDiT](https://github.com/NUS-HPC-AI-Lab/OpenDiT): An acceleration for DiT training. We adopt valuable acceleration
  strategies for training progress from OpenDiT.
* [PixArt](https://github.com/PixArt-alpha/PixArt-alpha): An open-source DiT-based text-to-image model.
* [Latte](https://github.com/Vchitect/Latte): An attempt to efficiently train DiT for video.
* [StabilityAI VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse-original): A powerful image VAE model.
* [CLIP](https://github.com/openai/CLIP): A powerful text-image embedding model.
* [T5](https://github.com/google-research/text-to-text-transfer-transformer): A powerful text encoder.
* [LLaVA](https://github.com/haotian-liu/LLaVA): A powerful image captioning model based on [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) and [Yi-34B](https://huggingface.co/01-ai/Yi-34B).

We are grateful for their exceptional work and generous contribution to open source.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=hpcaitech/Open-Sora&type=Date)](https://star-history.com/#hpcaitech/Open-Sora&Date)

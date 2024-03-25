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
</div>

## Open-Sora: Democratizing Efficient Video Production for All

We present **Open-Sora**, an initiative dedicated to **efficiently** produce high-quality video and make the model,
tools and contents accessible to all. By embracing **open-source** principles,
Open-Sora not only democratizes access to advanced video generation techniques, but also offers a
streamlined and user-friendly platform that simplifies the complexities of video production.
With Open-Sora, we aim to inspire innovation, creativity, and inclusivity in the realm of content creation.

[[中文文档]](/docs/zh_CN/README.md)

<h4>Open-Sora is still at an early stage and under active development.</h4>

## 📰 News

* **[2024.03.18]** 🔥 We release **Open-Sora 1.0**, a fully open-source project for video generation.
  Open-Sora 1.0 supports a full pipeline of video data preprocessing, training with
  <a href="https://github.com/hpcaitech/ColossalAI"><img src="assets/readme/colossal_ai.png" width="8%" ></a>
  acceleration,
  inference, and more. Our provided [checkpoints](#model-weights) can produce 2s 512x512 videos with only 3 days
  training.
  [[blog]](https://hpc-ai.com/blog/open-sora-v1.0)
* **[2024.03.04]** Open-Sora provides training with 46% cost reduction.
  [[blog]](https://hpc-ai.com/blog/open-sora)

## 🎥 Latest Demo

| **2s 512×512**                                                                                                                                                                 | **2s 512×512**                                                                                                                                                              | **2s 512×512**                                                                                                                                    |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| [<img src="assets/readme/sample_0.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/de1963d3-b43b-4e68-a670-bb821ebb6f80)                                 | [<img src="assets/readme/sample_1.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/13f8338f-3d42-4b71-8142-d234fbd746cc)                              | [<img src="assets/readme/sample_2.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/fa6a65a6-e32a-4d64-9a9e-eabb0ebb8c16)    |
| A serene night scene in a forested area. [...] The video is a time-lapse, capturing the transition from day to night, with the lake and forest serving as a constant backdrop. | A soaring drone footage captures the majestic beauty of a coastal cliff, [...] The water gently laps at the rock base and the greenery that clings to the top of the cliff. | The majestic beauty of a waterfall cascading down a cliff into a serene lake. [...] The camera angle provides a bird's eye view of the waterfall. |
| [<img src="assets/readme/sample_3.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/64232f84-1b36-4750-a6c0-3e610fa9aa94)                                 | [<img src="assets/readme/sample_4.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/983a1965-a374-41a7-a76b-c07941a6c1e9)                              | [<img src="assets/readme/sample_5.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/ec10c879-9767-4c31-865f-2e8d6cf11e65)    |
| A bustling city street at night, filled with the glow of car headlights and the ambient light of streetlights. [...]                                                           | The vibrant beauty of a sunflower field. The sunflowers are arranged in neat rows, creating a sense of order and symmetry. [...]                                            | A serene underwater scene featuring a sea turtle swimming through a coral reef. The turtle, with its greenish-brown shell [...]                   |

Videos are downsampled to `.gif` for display. Click for original videos. Prompts are trimmed for display,
see [here](/assets/texts/t2v_samples.txt) for full prompts. See more samples at
our [gallery](https://hpcaitech.github.io/Open-Sora/).

## 🔆 New Features/Updates

* 📍 Open-Sora-v1 released. Model weights are available [here](#model-weights). With only 400K video clips and 200 H800
  days (compared with 152M samples in Stable Video Diffusion), we are able to generate 2s 512×512 videos.
* ✅ Three stages training from an image diffusion model to a video diffusion model. We provide the weights for each
  stage.
* ✅ Support training acceleration including accelerated transformer, faster T5 and VAE, and sequence parallelism.
  Open-Sora improve **55%** training speed when training on 64x512x512 videos. Details locates
  at [acceleration.md](docs/acceleration.md).
* ✅ We provide data preprocessing pipeline,
  including [downloading](/tools/datasets/README.md), [video cutting](/tools/scenedetect/README.md),
  and [captioning](/tools/caption/README.md) tools. Our data collection plan can be found
  at [datasets.md](docs/datasets.md).
* ✅ We find VQ-VAE from [VideoGPT](https://wilson1yan.github.io/videogpt/index.html) has a low quality and thus adopt a
  better VAE from [Stability-AI](https://huggingface.co/stabilityai/sd-vae-ft-mse-original). We also find patching in
  the time dimension deteriorates the quality. See our **[report](docs/report_v1.md)** for more discussions.
* ✅ We investigate different architectures including DiT, Latte, and our proposed STDiT. Our **STDiT** achieves a better
  trade-off between quality and speed. See our **[report](docs/report_v1.md)** for more discussions.
* ✅ Support clip and T5 text conditioning.
* ✅ By viewing images as one-frame videos, our project supports training DiT on both images and videos (e.g., ImageNet &
  UCF101). See [commands.md](docs/commands.md) for more instructions.
* ✅ Support inference with official weights
  from [DiT](https://github.com/facebookresearch/DiT), [Latte](https://github.com/Vchitect/Latte),
  and [PixArt](https://pixart-alpha.github.io/).

<details>
<summary>View more</summary>

* ✅ Refactor the codebase. See [structure.md](docs/structure.md) to learn the project structure and how to use the
  config files.

</details>

### TODO list sorted by priority

* [ ] Complete the data processing pipeline (including dense optical flow, aesthetics scores, text-image similarity,
  deduplication, etc.). See [datasets.md](/docs/datasets.md) for more information. **[WIP]**
* [ ] Training Video-VAE. **[WIP]**

<details>
<summary>View more</summary>

* [ ] Support image and video conditioning.
* [ ] Evaluation pipeline.
* [ ] Incoporate a better scheduler, e.g., rectified flow in SD3.
* [ ] Support variable aspect ratios, resolutions, durations.
* [ ] Support SD3 when released.

</details>

## Contents

* [Installation](#installation)
* [Model Weights](#model-weights)
* [Inference](#inference)
* [Data Processing](#data-processing)
* [Training](#training)
* [Contribution](#contribution)
* [Acknowledgement](#acknowledgement)
* [Citation](#citation)

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
pip install packaging ninja
pip install flash-attn --no-build-isolation

# install apex (optional)
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

# install xformers
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121

# install this project
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora
pip install -v .
```

After installation, we suggest reading [structure.md](docs/structure.md) to learn the project structure and how to use
the config files.

## Model Weights

| Resolution | Data   | #iterations | Batch Size | GPU days (H800) | URL                                                                                           |
|------------|--------|-------------|------------|-----------------|-----------------------------------------------------------------------------------------------|
| 16×512×512 | 20K HQ | 20k         | 2×64       | 35              | [:link:](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x512x512.pth) |
| 16×256×256 | 20K HQ | 24k         | 8×64       | 45              | [:link:](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x256x256.pth) |
| 16×256×256 | 366K   | 80k         | 8×64       | 117             | [:link:](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-16x256x256.pth)    |

Our model's weight is partially initialized from [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha). The number of
parameters is 724M. More information about training can be found in our **[report](/docs/report_v1.md)**. More about
the dataset can be found in [datasets.md](/docs/datasets.md). HQ means high quality.

:warning: **LIMITATION**: Our model is trained on a limited budget. The quality and text alignment is relatively poor.
The model performs badly, especially on generating human beings and cannot follow detailed instructions. We are working
on improving the quality and text alignment.

## Inference

We have provided a Gradio application in this repository, you can use the following the command to start an interactive web application to experience video generation with Open-Sora.

```bash
pip install gradio
python scripts/demo.py
```

This will launch a Gradio application on your localhost.

Besides, we have also provided an offline inference script. Run the following commands to generate samples, the required model weights will be automatically downloaded. To change sampling prompts, modify the txt file passed to `--prompt-path`. See [here](docs/structure.md#inference-config-demos) to customize the configuration.

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

## Data Processing

High-quality Data is the key to high-quality models. Our used datasets and data collection plan
is [here](/docs/datasets.md). We provide tools to process video data. Currently, our data processing pipeline includes
the following steps:

1. Downloading datasets. [[docs](/tools/datasets/README.md)]
2. Split videos into clips. [[docs](/tools/scenedetect/README.md)]
3. Generate video captions. [[docs](/tools/caption/README.md)]

## Training

To launch training, first download [T5](https://huggingface.co/DeepFloyd/t5-v1_1-xxl/tree/main) weights
into `pretrained_models/t5_ckpts/t5-v1_1-xxl`. Then run the following commands to launch training on a single node.

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
* [LLaVA](https://github.com/haotian-liu/LLaVA): A powerful image captioning model based
  on [Yi-34B](https://huggingface.co/01-ai/Yi-34B).

We are grateful for their exceptional work and generous contribution to open source.

## Citation

```bibtex
@software{opensora,
  author = {Zangwei Zheng and Xiangyu Peng and Yang You},
  title = {Open-Sora: Democratizing Efficient Video Production for All},
  month = {March},
  year = {2024},
  url = {https://github.com/hpcaitech/Open-Sora}
}
```

[Zangwei Zheng](https://github.com/zhengzangw) and [Xiangyu Peng](https://github.com/xyupeng) equally contributed to
this work during their internship at [HPC-AI Tech](https://hpc-ai.com/).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=hpcaitech/Open-Sora&type=Date)](https://star-history.com/#hpcaitech/Open-Sora&Date)

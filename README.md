<p align="center">
    <img src="./assets/readme/icon_new.png" width="250"/>
<p>

</p>
<div align="center">
    <a href="https://github.com/hpcaitech/Open-Sora/stargazers"><img src="https://img.shields.io/github/stars/hpcaitech/Open-Sora?style=social"></a>
    <a href="https://github.com/hpcaitech/public_assets/tree/main/colossalai/contact/slack"><img src="https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp"></a>
    <a href="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png"><img src="https://img.shields.io/badge/微信-加入-green?logo=wechat&amp"></a>
</div>

## Open-Sora: Towards Open Reproduction of Sora

**Open-Sora** is an **open-source** initiative dedicated to **efficiently** reproducing OpenAI's Sora. Our project aims to cover **the full pipeline**, including video data preprocessing, training with acceleration, efficient inference and more. Operating on a limited budget, we prioritize the vibrant open-source community, providing access to text-to-image, image captioning, and language models. We hope to make a contribution to the community and make the project more accessible to everyone.

## 📰 News

* **[2024.03.18]** 🔥 We release **Open-Sora 1.0**, an open-source project to reproduce OpenAI Sora.
Open-Sora 1.0 supports a full pipeline of video data preprocessing, training with
<a href="https://github.com/hpcaitech/ColossalAI"><img src="assets/readme/colossal_ai.png" width="8%" ></a> acceleration,
inference, and more. Our provided checkpoint can produce 2s 512x512 videos.


## 🎥 Latest Demo

| **2s 512x512**                                  | **2s 512x512**                                  | **2s 512x512**                                 |
| ----------------------------------------------- | ----------------------------------------------- |-------------------------------------------------|
| [<img src="assets/readme/sample_0.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/de1963d3-b43b-4e68-a670-bb821ebb6f80) | [<img src="assets/readme/sample_1.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/13f8338f-3d42-4b71-8142-d234fbd746cc) | <img src="assets/readme/sample_2.gif" width=""> |

Click for the original video.

## 🔆 New Features/Updates

- 📍 Open-Sora-v1 is trained on xxx. We train the model in three stages. Model weights are available here. Training details can be found here. [WIP]
- ✅ Support training acceleration including flash-attention, accelerated T5, mixed precision, gradient checkpointing, splitted VAE, sequence parallelism, etc. XXX times. Details locates at [acceleration.md](docs/acceleration.md). [WIP]
- ✅ We provide video cutting and captioning tools for data preprocessing. Instructions can be found [here](tools/data/README.md) and our data collection plan can be found at [datasets.md](docs/datasets.md).
- ✅ We find VQ-VAE from [VideoGPT](https://wilson1yan.github.io/videogpt/index.html) has a low quality and thus adopt a better VAE from [Stability-AI](https://huggingface.co/stabilityai/sd-vae-ft-mse-original). We also find patching in the time dimension deteriorates the quality. See our **[report](docs/report_v1.md)** for more discussions.
- ✅ We investigate different architectures including DiT, Latte, and our proposed STDiT. Our **STDiT** achieves a better trade-off between quality and speed. See our **[report](docs/report_v1.md)** for more discussions.
- ✅ Support clip and T5 text conditioning.
- ✅ By viewing images as one-frame videos, our project supports training DiT on both images and videos (e.g., ImageNet & UCF101). See [command.md](docs/command.md) for more instructions.
- ✅ Support inference with official weights from [DiT](https://github.com/facebookresearch/DiT), [Latte](https://github.com/Vchitect/Latte), and [PixArt](https://pixart-alpha.github.io/).

<details>
<summary>View more</summary>

- ✅ Refactor the codebase. See [structure.md](docs/structure.md) to learn the project structure and how to use the config files.

</details>

### TODO list sorted by priority

- [ ] Complete the data processing pipeline (including dense optical flow, aesthetics scores, text-image similarity, deduplication, etc.). See [datasets.md]() for more information. **[WIP]**
- [ ] Training Video-VAE. **[WIP]**

<details>
<summary>View more</summary>

- [ ] Support image and video conditioning.
- [ ] Evaluation pipeline.
- [ ] Incoporate a better scheduler, e.g., rectified flow in SD3.
- [ ] Support variable aspect ratios, resolutions, durations.
- [ ] Support SD3 when released.

</details>

## Contents

- [Open-Sora: Towards Open Reproduction of Sora](#open-sora-towards-open-reproduction-of-sora)
- [📰 News](#-news)
- [🎥 Latest Demo](#-latest-demo)
- [🔆 New Features/Updates](#-new-featuresupdates)
  - [TODO list sorted by priority](#todo-list-sorted-by-priority)
- [Contents](#contents)
- [Installation](#installation)
- [Model Weights](#model-weights)
- [Inference](#inference)
- [Data Processing](#data-processing)
  - [Split video into clips](#split-video-into-clips)
  - [Generate video caption](#generate-video-caption)
- [Training](#training)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [Star History](#star-history)
- [TODO](#todo)

## Installation

```bash
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora
pip install xxx
```

After installation, we suggest reading [structure.md](docs/structure.md) to learn the project structure and how to use the config files.

## Model Weights

| Model      | #Params | url |
| ---------- | ------- | --- |
| 16x256x256 |         |     |

## Inference

```bash
python scripts/inference.py configs/opensora/inference/16x256x256.py
```

## Data Processing

### Split video into clips

We provide code to split a long video into separate clips efficiently using `multiprocessing`. See `tools/data/scene_detect.py`.

### Generate video caption

## Training

## Acknowledgement

* [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers.
* [OpenDiT](https://github.com/NUS-HPC-AI-Lab/OpenDiT): An acceleration for DiT training. OpenDiT's team provides valuable suggestions on acceleration of our training process.
* [PixArt](https://github.com/PixArt-alpha/PixArt-alpha): An open-source DiT-based text-to-image model.
* [Latte](https://github.com/Vchitect/Latte): An attempt to efficiently train DiT for video.
* [StabilityAI VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse-original): A powerful image VAE model.
* [CLIP](https://github.com/openai/CLIP): A powerful text-image embedding model.
* [T5](https://github.com/google-research/text-to-text-transfer-transformer): A powerful text encoder.
* [LLaVA](https://github.com/haotian-liu/LLaVA): A powerful image captioning model based on [LLaMA](https://github.com/meta-llama/llama) and [Yi-34B](https://huggingface.co/01-ai/Yi-34B).
* [PySceneDetect](https://github.com/Breakthrough/PySceneDetect): A powerful tool to split video into clips.

We are grateful for their exceptional work and generous contribution to open source.

## Citation

```bibtex
@software{opensora,
  author = {Zangwei Zheng and Xiangyu Peng and Shenggui Li and Yang You},
  title = {Open-Sora: Towards Open Reproduction of Sora},
  month = {March},
  year = {2024},
  url = {https://github.com/hpcaitech/Open-Sora}
}
```

[Zangwei Zheng](https://github.com/zhengzangw) and [Xiangyu Peng](https://github.com/xyupeng) equally contributed to this work during their internship at [HPC-AI Tech](https://hpc-ai.com/).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=hpcaitech/Open-Sora&type=Date)](https://star-history.com/#hpcaitech/Open-Sora&Date)

## TODO

Modules for releasing:

* `configs`
* `opensora`
* `assets`
* `scripts`
* `tools`

packages for data processing

put all outputs under ./checkpoints/, including pretrained_models, checkpoints, samples

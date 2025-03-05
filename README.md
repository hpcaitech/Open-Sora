<p align="center">
    <img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/readme/icon.png" width="250"/>
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

We design and implement **Open-Sora**, an initiative dedicated to **efficiently** producing high-quality video. We hope to make the model,
tools and all details accessible to all. By embracing **open-source** principles,
Open-Sora not only democratizes access to advanced video generation techniques, but also offers a
streamlined and user-friendly platform that simplifies the complexities of video generation.
With Open-Sora, our goal is to foster innovation, creativity, and inclusivity within the field of content creation.

🎬 For a professional and better version of the model, please try [Video Ocean](https://video-ocean.com/).
<div align="center">
   <a href="https://video-ocean.com/">
   <img src="https://github.com/hpcaitech/public_assets/blob/main/colossalai/img/3.gif" width="850" />
   </a>
</div>

[[中文文档](/docs/zh_CN/README.md)] [[潞晨云](https://cloud.luchentech.com/)|[OpenSora镜像](https://cloud.luchentech.com/doc/docs/image/open-sora/)|[视频教程](https://www.bilibili.com/video/BV1ow4m1e7PX/?vd_source=c6b752764cd36ff0e535a768e35d98d2)]

<div align="center">
   <a href="https://hpc-ai.com/?utm_source=github&utm_medium=social&utm_campaign=promotion-opensora">
   <img src="https://github.com/hpcaitech/public_assets/blob/main/colossalai/img/1.gif" width="850" />
   </a>
</div>

## 📰 News

- **[2025.03.17]** 🔥 We released **Open-Sora 2.0** (11B).
- **[2025.02.20]** 🔥 We released **Open-Sora 1.3** (1B). With the upgraded VAE and Transformer architecture, the quality of our generated videos has been greatly improved 🚀. [[checkpoints]](#open-sora-13-model-weights) [[report]](/docs/report_04.md) [[demo]](https://huggingface.co/spaces/hpcai-tech/open-sora)
- **[2024.12.23]** The development cost of video generation models has saved by 50%! Open-source solutions are now available with H200 GPU vouchers. [[blog]](https://company.hpc-ai.com/blog/the-development-cost-of-video-generation-models-has-saved-by-50-open-source-solutions-are-now-available-with-h200-gpu-vouchers) [[code]](https://github.com/hpcaitech/Open-Sora/blob/main/scripts/train.py) [[vouchers]](https://colossalai.org/zh-Hans/docs/get_started/bonus/)
- **[2024.06.17]** We released **Open-Sora 1.2**, which includes **3D-VAE**, **rectified flow**, and **score condition**. The video quality is greatly improved. [[checkpoints]](#open-sora-12-model-weights) [[report]](/docs/report_03.md)   [[blog]](https://hpc-ai.com/blog/open-sora-from-hpc-ai-tech-team-continues-open-source-generate-any-16-second-720p-hd-video-with-one-click-model-weights-ready-to-use)
- **[2024.04.25]** 🤗 We released the [Gradio demo for Open-Sora](https://huggingface.co/spaces/hpcai-tech/open-sora) on Hugging Face Spaces.
- **[2024.04.25]** We released **Open-Sora 1.1**, which supports **2s~15s, 144p to 720p, any aspect ratio** text-to-image, **text-to-video, image-to-video, video-to-video, infinite time** generation. In addition, a full video processing pipeline is released. [[checkpoints]](#open-sora-11-model-weights) [[report]](/docs/report_02.md)
- **[2024.03.18]** We released **Open-Sora 1.0**, a fully open-source project for video generation.
  Open-Sora 1.0 supports a full pipeline of video data preprocessing, training with
  <a href="https://github.com/hpcaitech/ColossalAI"><img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/readme/colossal_ai.png" width="8%" ></a>
  acceleration,
  inference, and more. Our model can produce 2s 512x512 videos with only 3 days training. [[checkpoints]](#open-sora-10-model-weights)
  [[blog]](https://hpc-ai.com/blog/open-sora-v1.0) [[report]](/docs/report_01.md)
- **[2024.03.04]** Open-Sora provides training with 46% cost reduction.
  [[blog]](https://hpc-ai.com/blog/open-sora)

📍 Since Open-Sora is under active development, we remain different branchs for different versions. The latest version is [main](https://github.com/hpcaitech/Open-Sora). Old versions include: [v1.0](https://github.com/hpcaitech/Open-Sora/tree/opensora/v1.0), [v1.1](https://github.com/hpcaitech/Open-Sora/tree/opensora/v1.1), [v1.2](https://github.com/hpcaitech/Open-Sora/tree/opensora/v1.2), [v1.3](https://github.com/hpcaitech/Open-Sora/tree/opensora/v1.3).

## 🎥 Latest Demo

More samples and corresponding prompts are available in our [Gallery](https://hpcaitech.github.io/Open-Sora/).

<details>
<summary>OpenSora 1.2 Demo</summary>

| **5s 720×1280**                                                                                                                                                        | **5s 720×1280**                                                                                                                                                           | **5s 720×1280**                                                                                                                                                              |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.3/demo_tomato.gif" width="">](https://streamable.com/e/r0imrp?quality=highest&amp;autoplay=1) | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.3/demo_fisherman.gif" width="">](https://streamable.com/e/hfvjkh?quality=highest&amp;autoplay=1) | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.3/demo_girl2.gif" width="">](https://streamable.com/e/kutmma?quality=highest&amp;autoplay=1)        |
| [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.3/demo_grape.gif" width="">](https://streamable.com/e/osn1la?quality=highest&amp;autoplay=1)  | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.3/demo_mushroom.gif" width="">](https://streamable.com/e/l1pzws?quality=highest&amp;autoplay=1)  | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.3/demo_parrot.gif" width="">](https://streamable.com/e/2vqari?quality=highest&amp;autoplay=1)       |
| [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.3/demo_trans.gif" width="">](https://streamable.com/e/1in7d6?quality=highest&amp;autoplay=1)  | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.3/demo_bear.gif" width="">](https://streamable.com/e/e9bi4o?quality=highest&amp;autoplay=1)      | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.3/demo_futureflower.gif" width="">](https://streamable.com/e/09z7xi?quality=highest&amp;autoplay=1) |
| [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.3/demo_fire.gif" width="">](https://streamable.com/e/16c3hk?quality=highest&amp;autoplay=1)   | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.3/demo_man.gif" width="">](https://streamable.com/e/wi250w?quality=highest&amp;autoplay=1)       | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.3/demo_black.gif" width="">](https://streamable.com/e/vw5b64?quality=highest&amp;autoplay=1)        |

</details>

<details>
<summary>OpenSora 1.2 Demo</summary>

| **4s 720×1280**                                                                                                                                                                                     | **4s 720×1280**                                                                                                                                                                                     | **4s 720×1280**                                                                                                                                                                                     |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.2/sample_0013.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/7895aab6-ed23-488c-8486-091480c26327) | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.2/sample_1718.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/20f07c7b-182b-4562-bbee-f1df74c86c9a) | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.2/sample_0087.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/3d897e0d-dc21-453a-b911-b3bda838acc2) |
| [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.2/sample_0052.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/644bf938-96ce-44aa-b797-b3c0b513d64c) | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.2/sample_1719.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/272d88ac-4b4a-484d-a665-8d07431671d0) | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.2/sample_0002.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/ebbac621-c34e-4bb4-9543-1c34f8989764) |
| [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.2/sample_0011.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/a1e3a1a3-4abd-45f5-8df2-6cced69da4ca) | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.2/sample_0004.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/d6ce9c13-28e1-4dff-9644-cc01f5f11926) | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.2/sample_0061.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/561978f8-f1b0-4f4d-ae7b-45bec9001b4a) |

</details>

<details>
<summary>OpenSora 1.1 Demo</summary>

| **2s 240×426**                                                                                                                                                                                                  | **2s 240×426**                                                                                                                                                                                                 |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.1/sample_16x240x426_9.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/c31ebc52-de39-4a4e-9b1e-9211d45e05b2) | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.1/sora_16x240x426_26.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/c31ebc52-de39-4a4e-9b1e-9211d45e05b2) |
| [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.1/sora_16x240x426_27.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/f7ce4aaa-528f-40a8-be7a-72e61eaacbbd)  | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.1/sora_16x240x426_40.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/5d58d71e-1fda-4d90-9ad3-5f2f7b75c6a9) |

| **2s 426×240**                                                                                                                                                                                                 | **4s 480×854**                                                                                                                                                                                                  |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.1/sora_16x426x240_24.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/34ecb4a0-4eef-4286-ad4c-8e3a87e5a9fd) | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.1/sample_32x480x854_9.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/c1619333-25d7-42ba-a91c-18dbc1870b18) |

| **16s 320×320**                                                                                                                                                                                            | **16s 224×448**                                                                                                                                                                                            | **2s 426×240**                                                                                                                                                                                                |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.1/sample_16s_320x320.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/3cab536e-9b43-4b33-8da8-a0f9cf842ff2) | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.1/sample_16s_224x448.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/9fb0b9e0-c6f4-4935-b29e-4cac10b373c4) | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.1/sora_16x426x240_3.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/3e892ad2-9543-4049-b005-643a4c1bf3bf) |

</details>

<details>
<summary>OpenSora 1.0 Demo</summary>

| **2s 512×512**                                                                                                                                                                                   | **2s 512×512**                                                                                                                                                                                   | **2s 512×512**                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.0/sample_0.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/de1963d3-b43b-4e68-a670-bb821ebb6f80) | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.0/sample_1.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/13f8338f-3d42-4b71-8142-d234fbd746cc) | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.0/sample_2.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/fa6a65a6-e32a-4d64-9a9e-eabb0ebb8c16) |
| A serene night scene in a forested area. [...] The video is a time-lapse, capturing the transition from day to night, with the lake and forest serving as a constant backdrop.                   | A soaring drone footage captures the majestic beauty of a coastal cliff, [...] The water gently laps at the rock base and the greenery that clings to the top of the cliff.                      | The majestic beauty of a waterfall cascading down a cliff into a serene lake. [...] The camera angle provides a bird's eye view of the waterfall.                                                |
| [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.0/sample_3.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/64232f84-1b36-4750-a6c0-3e610fa9aa94) | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.0/sample_4.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/983a1965-a374-41a7-a76b-c07941a6c1e9) | [<img src="https://github.com/hpcaitech/Open-Sora-Demo/blob/main/demo/v1.0/sample_5.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/ec10c879-9767-4c31-865f-2e8d6cf11e65) |
| A bustling city street at night, filled with the glow of car headlights and the ambient light of streetlights. [...]                                                                             | The vibrant beauty of a sunflower field. The sunflowers are arranged in neat rows, creating a sense of order and symmetry. [...]                                                                 | A serene underwater scene featuring a sea turtle swimming through a coral reef. The turtle, with its greenish-brown shell [...]                                                                  |

Videos are downsampled to `.gif` for display. Click for original videos. Prompts are trimmed for display,
see [here](/assets/texts/t2v_samples.txt) for full prompts.

</details>

## 🔆 Reports

- **[Tech Report of Open-Sora 2.0]()**
- **[Step by step to train or finetune your own model](docs/train.md)**
- **[Step by step to train and evaluate an video autoencoder](docs/ae.md)**
- Reports of previous version (better see in according branch):
  - Open-Sora 1.3: [report 1.3](docs/report_04.md): shift-window attention, unified spatial-temporal VAE, etc.
  - Open-Sora 1.2: [Tech Report](https://arxiv.org/abs/2412.20404), [report 1.2](docs/report_03.md): rectified flow, 3d-VAE, score condition, evaluation, etc.
  - Open-Sora 1.1: [report 1.1](docs/report_02.md): multi-resolution/length/aspect-ratio, image/video conditioning/editing, data preprocessing, etc.
  - Open-Sora 1.0: [report 1.0](docs/report_01.md): architecture, captioning, etc.
- Gallery: [gallery](https://hpcaitech.github.io/Open-Sora/)

📍 Since Open-Sora is under active development, we remain different branchs for different versions. The latest version is [main](https://github.com/hpcaitech/Open-Sora). Old versions include: [v1.0](https://github.com/hpcaitech/Open-Sora/tree/opensora/v1.0), [v1.1](https://github.com/hpcaitech/Open-Sora/tree/opensora/v1.1), [v1.2](https://github.com/hpcaitech/Open-Sora/tree/opensora/v1.2), [v1.3](https://github.com/hpcaitech/Open-Sora/tree/opensora/v1.3).

## Quickstart

### Installation

```bash
# create a virtual env and activate (conda as an example)
conda create -n opensora python=3.10
conda activate opensora

# download the repo
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora

# Ensure torch >= 2.4.0
pip install -v . # for development mode, `pip install -v -e .`
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 # install xformers according to your cuda version
pip install flash-attn --no-build-isolation
```

Optionally, you can install flash attention 3 for faster speed.

```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention/hopper
python setup.py install
```

### Model Download

Our 11B model supports 256px and 768px resolution. Both T2V and I2V are supported by one model. 🤗 [Huggingface](https://huggingface.co/hpcai-tech/Open-Sora-v2) 🤖 [ModelScope](https://modelscope.cn/models/luchentech/Open-Sora-v2).

Download from huggingface:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts
```

Download from ModelScope:

```bash
pip install modelscope
modelscope download hpcai-tech/Open-Sora-v2 --local_dir ./ckpts
```

## Contribution

Thanks goes to these wonderful contributors:

<a href="https://github.com/hpcaitech/Open-Sora/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=hpcaitech/Open-Sora" />
</a>

If you wish to contribute to this project, please refer to the [Contribution Guideline](./CONTRIBUTING.md).

### Text-to-Video Generation

Our model is optimi

## Acknowledgement

Here we only list a few of the projects. For other works and datasets, please refer to our report.

- [ColossalAI](https://github.com/hpcaitech/ColossalAI): A powerful large model parallel acceleration and optimization
  system.
- [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers.
- [OpenDiT](https://github.com/NUS-HPC-AI-Lab/OpenDiT): An acceleration for DiT training. We adopt valuable acceleration
  strategies for training progress from OpenDiT.
- [PixArt](https://github.com/PixArt-alpha/PixArt-alpha): An open-source DiT-based text-to-image model.
- [Flux](https://github.com/black-forest-labs/flux): A powerful text-to-image generation model.
- [Latte](https://github.com/Vchitect/Latte): An attempt to efficiently train DiT for video.
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo/tree/main?tab=readme-ov-file): Open-Source text-to-video model.
- [StabilityAI VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse-original): A powerful image VAE model.
- [DC-AE](https://github.com/mit-han-lab/efficientvit): Deep Compression AutoEncoder for image compression.
- [CLIP](https://github.com/openai/CLIP): A powerful text-image embedding model.
- [T5](https://github.com/google-research/text-to-text-transfer-transformer): A powerful text encoder.
- [LLaVA](https://github.com/haotian-liu/LLaVA): A powerful image captioning model based on [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) and [Yi-34B](https://huggingface.co/01-ai/Yi-34B).
- [PLLaVA](https://github.com/magic-research/PLLaVA): A powerful video captioning model.
- [MiraData](https://github.com/mira-space/MiraData): A large-scale video dataset with long durations and structured caption.

We are grateful for their exceptional work and generous contribution to open source. Special thanks go to the authors of [MiraData](https://github.com/mira-space/MiraData) and [Rectified Flow](https://github.com/gnobitab/RectifiedFlow) for their valuable advice and help. We wish to express gratitude towards AK for sharing this project on social media and Hugging Face for providing free GPU resources for our online Gradio demo.

## Citation

```bibtex
@software{opensora,
  author = {Zangwei Zheng and Xiangyu Peng and Tianji Yang and Chenhui Shen and Shenggui Li and Hongxin Liu and Yukun Zhou and Tianyi Li and Yang You},
  title = {Open-Sora: Democratizing Efficient Video Production for All},
  month = {March},
  year = {2024},
  url = {https://github.com/hpcaitech/Open-Sora}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=hpcaitech/Open-Sora&type=Date)](https://star-history.com/#hpcaitech/Open-Sora&Date)

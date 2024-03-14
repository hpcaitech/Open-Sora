# NAME TO BE ADDED

Speed at [docs/speed.md](docs/speed.md).

## Installation

```bash
# create a virtual env
conda create -n minisora python=3.9

# install torch
# the command below is for CUDA 12.1, choose install commands from
# https://pytorch.org/get-started/locally/ based on your own CUDA version
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# install flash attention (optional)
pip install packaging ninja
pip install flash-attn --no-build-isolation

# install apex (optional)
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

# install this project
pip install -v .

# if you are in development mode
pip install -v -e .
```

> For now, you can use `/home/zhaowangbo/zangwei/minisora_new/.conda/envs/dit` as the conda environment.

## Usage

The basic usage is as follows:

```bash
# sample
python nanosora/sample.py configs/sample/IN-pixart-official.py

# train on a single node
torchrun --standalone --nproc_per_node=8 nanosora/train.py ./configs/train/IN-pixart.py

# train with wandb
WANDB_API_KEY=$YOUR_WANDB_KEY torchrun --nnodes=1 --nproc_per_node=8 nanosora/train.py ./configs/train/IN-dit.py --cfg-options wandb=True
# Multi-node
colossalai run --nproc_per_node 8 --hostfile hostfile nanosora/train.py ./configs/train/ucf-pixart-st.py
```

The configs are located in `configs/sample` and `configs/train` respectively.

| Script                  | Description                                        |
| ----------------------- | -------------------------------------------------- |
| `IN-dit-official.py`    | Sampling with official DiT weight                  |
| `IN-dit-clip.py`        | Align official DiT weights with CLIP               |
| `IN-dit.py`             | Train DiT from scratch                             |
| `IN-pixart-official.py` | Sampling with official PixArt weight               |
| `IN-pixart.py`          | Train Pixart with official Pixart-256x256          |
| `ucf-dit.py`            | Train DiT on UCF101                                |
| `ucf-latte-official.py` | Sampling with official Latte weight                |
| `ucf-latte.py`          | Train Latte on UCF101                              |
| `ucf-dit-st.py`         | Train DiT on UCF101 with spatio-temporal attention |

## Features and TODOs

### Model

- [x] Support our own model with time-spatial attention and training with efficient re-parameterization.
- [x] Support training all models on images (e.g. ImageNet) and videos (e.g. UCF101).
- [x] Support official weights for [DiT](https://github.com/facebookresearch/DiT/tree/main), [Latte](https://github.com/Vchitect/Latte), and [PixArt](https://github.com/PixArt-alpha/PixArt-alpha).
- [x] Support [CLIP](https://github.com/openai/CLIP/tree/main) and [T5](https://github.com/google-research/text-to-text-transfer-transformer) condition.
- [x] Support patch embedding on time dimension, and treat image as one-frame video.
- [x] Support [Stable Diffusion](https://github.com/CompVis/latent-diffusion) VAE. We do not adopt [VideoVGT](https://github.com/wilson1yan/VideoGPT)'s VQ-VAE as the performance is poor.

**Variable Resolution**

- [ ] Support dynamic aspect ratios loading.fl
- [x] Support variable aspect ratios, resolutions, and frame numbers.

**High Performance Training**

**Training plan**

**Inference and Evaluation**

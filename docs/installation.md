# Installation

Requirements are listed in `requirements` folder.

## Different CUDA versions

You need to mannually install `torch`, `torchvision` and `xformers` for different CUDA versions.

```bash
# install torch (>=2.1 is recommended)
# the command below is for CUDA 12.1, choose install commands from
# https://pytorch.org/get-started/locally/ based on your own CUDA version
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# install xformers
# the command below is for CUDA 12.1, choose install commands from
# https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers based on your own CUDA version
pip install xformers --index-url https://download.pytorch.org/whl/cu121
```

```bash
# install flash attention (optional)
# set enable_flash_attn=False in config to avoid using flash attention
pip install packaging ninja
pip install flash-attn --no-build-isolation

# install apex (optional)
# set enable_layernorm_kernel=False in config to avoid using apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git
```

gdown
pre-commit
pyarrow
tensorboard
transformers
wandb
pandarallel
gradio
spaces

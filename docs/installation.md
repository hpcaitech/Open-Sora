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

## Different Dependencies

The default installation is for inference only. Other optional dependencies are listed below.

```bash
pip install -v .[data]  # for data preprocessing
pip install -v .[eval]  # for evaluation
```

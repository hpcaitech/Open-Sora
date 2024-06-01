# Installation

Requirements are listed in `requirements` folder.
Note that besides these packages, some packages needs to be mannually installed, and are detailed in the following sections.

## Different CUDA versions

You need to manually install `torch`, `torchvision` and `xformers` for different CUDA versions.

For CUDA 12.1,

```bash
pip install -U pip
# update, otherwise may run into weird issues with apex
pip install -U setuptools
pip install -U wheel

# install pytorch, torchvision, and xformers
pip install -r requirements/requirements-cu121.txt
```

For other CUDA versions, you need to manually install the corresponding packages.

- PyTorch: choose install commands from [PyTorch installation page](https://pytorch.org/get-started/locally/) based on your own CUDA version.
- xformers: choose install commands from [xformers repo](https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers) based on your own CUDA version.

Then, you can install the project for training and inference with the following commands:

```bash
# install this project
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora

# the default installation is for inference only
pip install -v . # NOTE: for development mode, run `pip install -v -e .`
```

(Optional, recommended for fast speed, especially for training) To enable `layernorm_kernel` and `flash_attn`, you need to install `apex` and `flash-attn` with the following commands.

```bash
# install flash attention
# set enable_flash_attn=False in config to disable flash attention
pip install packaging ninja
pip install flash-attn --no-build-isolation


# install apex, the compilation will take a long time
# set enable_layernorm_kernel=False in config to disable apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git
```

## Optional Dependencies

The default installation is for inference only. Other optional dependencies are detailed in sections below, namel "Data Dependencies" and "Evaluation Dependencies".

### Data Dependencies

First, run the following command to install requirements:

```bash
pip install -v .[data]
# For development: `pip install -v -e .[eval]`
```

Next, you need to manually install the packages listed in the following sections specific to your data processing needs.

#### OpenCV

To get image and video information, we use [opencv-python](https://github.com/opencv/opencv-python). You can install it with pip:

```bash
pip install opencv-python
```

However, if your videos are in av1 codec instead of h264, you need to install ffmpeg (already in our [requirement script](../requirements/requirements-data.txt)), then run the following to make conda support av1 codec:

```bash
pip uninstall opencv-python
conda install -c conda-forge opencv
```

#### LLaVA Captioning

You need to manually install LLaVA with the following command:

```bash
pip install --no-deps llava@git+https://github.com/haotian-liu/LLaVA.git@v1.2.2.post1
```

#### PLLaVA Captioning

You need to manually install PLLaVa with the following commands:

```bash
cd tools/caption/pllava_dir # Assume you are in Open-Sora-dev root directory
git clone https://github.com/magic-research/PLLaVA.git
cd PLLaVA
git checkout fd9194a # since there is no version tag, we use this commit
python python_scripts/hf.py # download the PLLaVA weights

# IMPORTANT: create new environment for reliable pllava performances:
conda create -n pllava python=3.10
# You need to manually install `torch`, `torchvision` and `xformers` for different CUDA versions, the following works for CUDA 12.1:
conda activate pllava
pip install -r ../../../requirements/requirements-cu121.txt
pip install packaging ninja
pip install flash-attn --no-build-isolation
# You may manually remove any lines in requirements.txt that contains `cu11`, then run `pip install -r requirements.txt`
# Alternatively, use our prepared pllava environment:
pip install -r ../../../../requirements/requirements-pllava.txt
```

#### Scene Detection

We use [`PySceneDetect`](https://github.com/Breakthrough/PySceneDetect) for this job. You need to manually run the following:

```bash
pip install scenedetect[opencv] --upgrade
```

#### OCR

You need to go into `path_to_your_env/lib/python3.10/site-packages/mmdet/__init__.py`
and change the assert of `mmcv_version < digit_version(mmcv_maximum_version)` to `mmcv_version <= digit_version(mmcv_maximum_version)`.

If you are unsure of your path to the mmdet init file, simply run our [OCR command](../tools/scoring/README.md), wait for the mmdeet assertion error on mmcv versions.
The error will contain the exact path to the mmdet init file.

### Evaluation Dependencies

First, run the following command to install requirements:

```bash
pip install -v .[eval]
# For development:`pip install -v -e .[eval]`
```

Next, you need to manually install the packages listed in the following sections specific to different evaluation methods.

#### VBench

You need to manually install [VBench](https://github.com/Vchitect/VBench):

```bash
pip install --no-deps vbench==0.1.1
# If the installation shows a warning about the intalled vbench not in PATH, you need to add it by:
export PATH="/path/to/vbench:$PATH"
```

#### VAE

You need to mannually install [cupy](https://docs.cupy.dev/en/stable/install.html).

- For CUDA v11.2~11.8 (x86_64 / aarch64), `pip install cupy-cuda11x`
- For CUDA v12.x (x86_64 / aarch64), `pip install cupy-cuda12x`

Note that for VAE evaluation, you may run into error with `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`, in this case, you need to go to the corresponding file (`.../pytorchvideo/transforms/augmentations.py`) reporting this error, then change as following:

```python
# find the original line:
import torchvision.transforms.functional_tensor as F_t
# change to:
import torchvision.transforms._functional_tensor as F_t
```

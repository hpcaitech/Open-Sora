# Installation

Requirements are listed in `requirements` folder.
Note that besides these packages, some packages needs to be mannually installed, and are detailed in the following sections.

## Training & Inference

You need to install `opensora` for training and inference. You can follow the steps below for installation. We also provide guideline for different CUDA versions for compatiblity.

Please note that the default installation is for training and inference only. Other optional dependencies are detailed in the sections [Data Processing](#data-processing), [Evaluation](#evaluation), and [VAE](#vae) respectively.

### Step 1: Install PyTorch and xformers

First of all, make sure you have the latest build toolkit for Python.

```bash
# update build libs
pip install -U pip setuptools wheel
```

If you are using **CUDA 12.1**,  you can execute the command below to directly install PyTorch, torchvision and xformers.

```bash
# install pytorch, torchvision, and xformers
pip install -r requirements/requirements-cu121.txt
```

If you are using different CUDA versions, you need to manually install `torch`, `torchvision` and `xformers`. You can find the compatible distributions according to the links below.

- PyTorch: choose install commands from [PyTorch installation page](https://pytorch.org/get-started/locally/) based on your own CUDA version.
- xformers: choose install commands from [xformers repo](https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers) based on your own CUDA version.

### Step 2: Install Open-Sora

Then, you can install the project for training and inference with the following commands:

```bash
# install this project
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora

# the default installation is for inference only
pip install -v . # NOTE: for development mode, run `pip install -v -e .`
```

### Step 3: Install Acceleration Tools (Optional)

This is optional but recommended for faster speed, especially for training. To enable `layernorm_kernel` and `flash_attn`, you need to install `apex` and `flash-attn` with the following commands.

```bash
# install flash attention
# set enable_flash_attn=False in config to disable flash attention
pip install packaging ninja
pip install flash-attn --no-build-isolation

# install apex, the compilation will take a long time
# set enable_layernorm_kernel=False in config to disable apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git
```

## Data Processing

### Step 1: Install Requirements

First, run the following command to install requirements:

```bash
pip install -v .[data]
# For development: `pip install -v -e .[eval]`
```

Next, you need to manually install the packages listed in the following sections specific to your data processing needs.

### Step 2: Install OpenCV

To get image and video information, we use [opencv-python](https://github.com/opencv/opencv-python). You can install it with pip:

```bash
pip install opencv-python
```

However, if your videos are in av1 codec instead of h264, you need to install ffmpeg (already in our [requirement script](../requirements/requirements-data.txt)), then run the following to make conda support av1 codec:

```bash
pip uninstall opencv-python
conda install -c conda-forge opencv
```

### Step 3: Install Task-specific Dependencies

We have a variety of data processing pipelines, each requires its own dependencies. You can refer to the sections below to install dependencies according to your own needs.

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


## Evaluation

### Step 1: Install Requirements

To conduct evaluation, run the following command to install requirements:

```bash
pip install -v .[eval]
# For development:`pip install -v -e .[eval]`
```

### Step 2: Install VBench

<!-- You need to manually install [VBench](https://github.com/Vchitect/VBench):

```bash
pip install --no-deps vbench==0.1.1
# If the installation shows a warning about the intalled vbench not in PATH, you need to add it by:
export PATH="/path/to/vbench:$PATH"
``` -->

You need to install VBench mannually by:
```bash
# first clone their repo
cd .. # assume you are in the Open-Sora root folder, you may install at other location but make sure the soft link paths later are correct
git clone https://github.com/Vchitect/VBench.git
cd VBench
git checkout v0.1.2

# next, fix their hard-coded path isse
vim vbench2_beta_i2v/utils.py
# find `image_root` in the `load_i2v_dimension_info` function, change it to point to your appropriate image folder

# last, create softlinks
cd ../Open-Sora # or `cd ../Open-Sora-dev` for development
ln -s ../VBench/vbench vbench # you may need to change ../VBench/vbench to your corresponding path
ln -s ../VBench/vbench2_beta_i2v vbench2_beta_i2v # you may need to change ../VBench/vbench_beta_i2v to your corresponding path
# later you need to make sure to run evaluation from your Open-Sora folder, else vbench, vbench2_beta_i2v cannot be found
```


### Step 3: Install `cupy` for Potential VAE Errors

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




## VAE

### Step 1: Install Requirements

To train and evaluate your own VAE, run the following command to install requirements:

```bash
pip install -v .[vae]
# For development:`pip install -v -e .[vae]`
```

### Step 2: VAE Evaluation (`cupy` and Potential VAE Errors)

Refer to the [Evaluation's VAE section](#step-3-install-cupy-for-potential-vae-errors) above.

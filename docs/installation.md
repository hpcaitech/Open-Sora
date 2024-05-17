# Installation

Requirements are listed in `requirements` folder.

## Different CUDA versions

You need to manually install `torch`, `torchvision` and `xformers` for different CUDA versions.

```bash
# install torch (>=2.1 is recommended)
# the command below is for CUDA 12.1, choose install commands from
# https://pytorch.org/get-started/locally/ based on your own CUDA version
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# install xformers
# the command below is for CUDA 12.1, choose install commands from
# https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers based on your own CUDA version
pip install xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu121
```

## Different Dependencies

The default installation is for inference only. Other optional dependencies are listed below.

```bash
pip install -v .[data]  # for data preprocessing
pip install -v .[eval]  # for evaluation, need to manually install some packages detailed below

# Note: if in development mode, use the following commands instead:
pip install -v -e .[data]
pip install -v -e .[eval]
```


## Data Dependencies

First, run the following command to install requirements:
```bash
pip install -v .[data]  # For development: `pip install -v -e .[eval]`
```
Next, you need to manually install the packages listed in the following sections specific to your data processing needs.

### LLaVA Captioning
You need to manually install LLaVA with the following command:
```bash
pip install --no-deps llava@git+https://github.com/haotian-liu/LLaVA.git@v1.2.2.post1
```
### PLLaVA Captioning

You need to manually install PLLaVa with the following commands:
```bash
cd tools/caption/pllava_dir # Assume you are in Open-Sora-dev root directory
git clone https://github.com/magic-research/PLLaVA.git
cd PLLaVA
git checkout fd9194a # since there is no version tag, we use this commit
python python_scripts/hf.py # download the PLLaVA weights
```

## Evaluation Dependencies

First, run the following command to install requirements:
```bash
pip install -v .[eval] # For development:`pip install -v -e .[eval]`
```
Next, you need to manually install the packages listed in the following sections specific to different evaluation methods.

### VBench
You need to manually install [VBench](https://github.com/Vchitect/VBench):
```bash
pip install --no-deps vbench==0.1.1
# If the installation shows a warning about the intalled vbench not in PATH, you need to add it by:
export PATH="/path/to/vbench:$PATH"
```


### VAE

You need to mannually install [cupy](https://docs.cupy.dev/en/stable/install.html).
* For CUDA v11.2~11.8 (x86_64 / aarch64), `pip install cupy-cuda11x`
* For CUDA v12.x (x86_64 / aarch64), `pip install cupy-cuda12x`

Note that for VAE evaluation, you may run into error with `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`, in this case, you need to go to the corresponding file (`.../pytorchvideo/transforms/augmentations.py`) reporting this error, then change as following:
```yaml
# find the original line:
import torchvision.transforms.functional_tensor as F_t
# change to:
import torchvision.transforms._functional_tensor as F_t
```

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
pip install xformers --index-url https://download.pytorch.org/whl/cu121
```

## Different Dependencies

The default installation is for inference only. Other optional dependencies are listed below.

```bash
pip install -v .[data]  # for data preprocessing
pip install -v .[eval]  # for evaluation, need to manually install some packages detailed below
```


## Evaluation Dependencies

```bash
pip install -v .[eval]
```

#### VBench
You need to manually install [VBench](https://github.com/Vchitect/VBench):
```bash
pip install --no-deps vbench==0.1.1
# If the installation shows a warning about the intalled vbench not in PATH, you need to add it by:
export PATH="/path/to/vbench:$PATH"
```


#### VAE

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

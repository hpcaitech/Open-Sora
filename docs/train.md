# Step by step to train or finetune your own model

## Installation

Besides from the installation in the main page, you need to install the following packages:

```bash
pip install git+https://github.com/hpcaitech/TensorNVMe.git # requires cmake, for checkpoint saving
pip install pandarallel # for parallel processing
```

## Prepare dataset

The dataset should be presented in a `csv` or `parquet` file. To better illustrate the process, we will use a 45k [pexels dataset](https://huggingface.co/datasets/hpcai-tech/open-sora-pexels-45k) as an example. This dataset contains clipped, score filtered high-quality videos from [Pexels](https://www.pexels.com/).

First, download the dataset to your local machine:

```bash
mkdir datasets
cd datasets
# For Chinese users, export HF_ENDPOINT=https://hf-mirror.com to speed up the download
huggingface-cli download --repo-type dataset hpcai-tech/open-sora-pexels-45k --local-dir open-sora-pexels-45k # 250GB

cd open-sora-pexels-45k
cat tar/pexels_45k.tar.* > pexels_45k.tar
tar -xvf pexels_45k.tar
mv pexels_45k .. # make sure the path is Open-Sora/datasets/pexels_45k
```

There are three `csv` files provided:

- `pexels_45k.csv`: contains only path and text, which needs to be processed for training.
- `pexels_45k_necessary.csv`: contains necessary information for training.
- `pexels_45k_score.csv`: contains score information for each video. The 45k videos are filtered out based on the score. See tech report for more details.

If you want to use custom dataset, at least the following columns are required:

```csv
path,text,num_frames,height,width,aspect_ratio,resolution,fps
```

We provide a script to process the `pexels_45k.csv` to `pexels_45k_necessary.csv`:

```bash
# single process
python scripts/cnv/meta.py --input datasets/pexels_45k.csv --output datasets/pexels_45k_nec.csv --num_workers 0
# parallel process
python scripts/cnv/meta.py --input datasets/pexels_45k.csv --output datasets/pexels_45k_nec.csv --num_workers 64
```

> The process may take a while, depending on the number of videos in the dataset. The process is neccessary for training on arbitrary aspect ratio, resolution, and number of frames.

## Training

The command format to launch training is as follows:

```bash
torchrun --nproc_per_node 8 scripts/diffusion/train.py [path/to/config] --dataset.data-path [path/to/dataset] [override options]
```

For example, to train a model with stage 1 config from scratch using pexels dataset:

```bash
torchrun --nproc_per_node 8 scripts/diffusion/train.py configs/diffusion/train/stage1.py --dataset.data-path datasets/pexels_45k_necessary.csv
```

### Config

All configs are located in `configs/diffusion/train/`. The following rules are applied:

- `_base_ = ["config_to_inherit"]`: inherit from another config by mmengine's support. Variables are overwritten by the new config. Dictionary is merged if `_delete_` key is not present.
- command line arguments override the config file. For example, `--lr 1e-5` will override the `lr` in the config file. `--dataset.data-path datasets/pexels_45k_necessary.csv` will override the `data-path` value in the dictionary `dataset`.

The `bucket_config` is used to control different training stages. It is a dictionary of dictionaries. The tuple means (sampling probability, batch size). For example:

```python
bucket_config = {
    "256px": {
        1: (1.0, 45), # for 256px images, use 100% of the data with batch size 45
        33: (1.0, 12), # for 256px videos with no less than 33 frames, use 100% of the data with batch size 12
        65: (1.0, 6), # for 256px videos with no less than 65 frames, use 100% of the data with batch size 6
        97: (1.0, 4), # for 256px videos with no less than 97 frames, use 100% of the data with batch size 4
        129: (1.0, 3), # for 256px videos with no less than 129 frames, use 100% of the data with batch size 3
    },
    "768px": {
        1: (0.5, 13), # for 768px images, use 50% of the data with batch size 13
    },
    "1024px": {
        1: (0.5, 7), # for 1024px images, use 50% of the data with batch size 7
    },
}
```

We provide the following configs, the batch size is searched on H200 GPUs with 140GB memory:

- `image.py`: train on images only.
- `stage1.py`: train on videos with 256px resolution.
- `stage2.py`: train on videos with 768px resolution with sequence parallelism (default 4).
- `stage1_i2v.py`: train t2v and i2v with 256px resolution.
- `stage2_i2v.py`: train t2v and i2v with 768px resolution.

We also provide a demo config `demo.py` with small batch size for debugging.

### Fine-tuning

To finetune from Open-Sora v2, run:

```bash
torchrun --nproc_per_node 8 scripts/diffusion/train.py configs/diffusion/train/stage1.py --dataset.data-path datasets/pexels_45k_necessary.csv --model.from_pretrained ckpts/Open_Sora_v2.safetensors
```

To finetune from flux-dev, we provided a transformed flux-dev [ckpts](https://huggingface.co/hpcai-tech/flux1-dev-fused-rope). Download it to `ckpts` and run:

```bash
torchrun --nproc_per_node 8 scripts/diffusion/train.py configs/diffusion/train/stage1.py --dataset.data-path datasets/pexels_45k_necessary.csv --model.from_pretrained ckpts/flux1-dev-fused-rope.safetensors
```

### Multi-GPU

To train on multiple GPUs, use `colossalai run`:

```bash
colossalai run --hostfile hostfiles --nproc_per_node 8 scripts/diffusion/train.py configs/diffusion/train/stage1.py --dataset.data-path datasets/pexels_45k_necessary.csv --model.from_pretrained ckpts/Open_Sora_v2.safetensors
```

`hostfiles` is a file that contains the IP addresses of the nodes. For example:

```bash
xxx.xxx.xxx.xxx
yyy.yyy.yyy.yyy
zzz.zzz.zzz.zzz
```

use `--wandb True` to log the training process to [wandb](https://wandb.ai/).

### Resume training

To resume training, use `--load`. It will load the optimizer state and dataloader state.

```bash
torchrun --nproc_per_node 8 scripts/diffusion/train.py configs/diffusion/train/stage1.py --dataset.data-path datasets/pexels_45k_necessary.csv --load outputs/your_experiment/epoch*-global_step*
```

If you want to load optimzer state but not dataloader state, use:

```bash
torchrun --nproc_per_node 8 scripts/diffusion/train.py configs/diffusion/train/stage1.py --dataset.data-path datasets/pexels_45k_necessary.csv --load outputs/your_experiment/epoch*-global_step* --start-step 0 --start-epoch 0
```

> Note if dataset, batch size, and number of GPUs are changed, the dataloader state will not be meaningful.

## Inference

The inference is the same as described in the main page. The command format is as follows:

```bash
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py configs/diffusion/inference/t2i2v_256px.py --save-dir samples --prompt "raining, sea" --model.from_pretrained outputs/your_experiment/epoch*-global_step*
```

## Advanced Usage

More details are provided in the tech report. If explanation for some techiques is needed, feel free to open an issue.

- Tensor parallelism and sequence parallelism
- Zero 2
- Pin memory organization
- Garbage collection organization
- Data prefetching
- Communication bucket optimization
- Shardformer for T5

### Gradient Checkpointing

We support selective gradient checkpointing to save memory. The `grad_ckpt_setting` is a tuple, the first element is the number of dual layers to apply gradient checkpointing, the second element is the number of single layers to apply full gradient. A very large number will apply full gradient to all layers.

```python
grad_ckpt_setting = (100, 100)
model = dict(
    grad_ckpt_setting=grad_ckpt_setting,
)
```

To further save memory, you can offload gradient checkpointing to CPU by:

```python
grad_ckpt_buffer_size = 25 * 1024**3 # 25GB
```

### Asynchronous Checkpoint Saving

With `--async-io True`, the checkpoint will be saved asynchronously with the support of ColossalAI. This will save time for checkpoint saving.

### Dataset

With a very large dataset, the `csv` file or even `parquet` file may be too large to fit in memory. We provide a script to split the dataset into smaller chunks:

```bash
python scripts/cnv/shard.py /path/to/dataset.parquet
```

Then a folder with shards will be created. You can use the `--dataset.memory_efficient True` to load the dataset shard by shard.

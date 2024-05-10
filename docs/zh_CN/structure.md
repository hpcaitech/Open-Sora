# 代码仓库和配置文件结构

## 代码仓库结构

```plaintext
Open-Sora
├── README.md
├── docs
│   ├── acceleration.md            -> Acceleration & Speed benchmark
│   ├── command.md                 -> Commands for training & inference
│   ├── datasets.md                -> Datasets used in this project
│   ├── structure.md               -> This file
│   └── report_v1.md               -> Report for Open-Sora v1
├── scripts
│   ├── train.py                   -> diffusion training script
│   └── inference.py               -> Report for Open-Sora v1
├── configs                        -> Configs for training & inference
├── opensora
│   ├── __init__.py
│   ├── registry.py                -> Registry helper
│   ├── acceleration               -> Acceleration related code
│   ├── dataset                    -> Dataset related code
│   ├── models
│   │   ├── layers                 -> Common layers
│   │   ├── vae                    -> VAE as image encoder
│   │   ├── text_encoder           -> Text encoder
│   │   │   ├── classes.py         -> Class id encoder (inference only)
│   │   │   ├── clip.py            -> CLIP encoder
│   │   │   └── t5.py              -> T5 encoder
│   │   ├── dit
│   │   ├── latte
│   │   ├── pixart
│   │   └── stdit                  -> Our STDiT related code
│   ├── schedulers                 -> Diffusion schedulers
│   │   ├── iddpm                  -> IDDPM for training and inference
│   │   └── dpms                   -> DPM-Solver for fast inference
│   └── utils
└── tools                          -> Tools for data processing and more
```

## 配置文件结构


我们的配置文件遵循[MMEgine](https://github.com/open-mmlab/mmengine)。 MMEngine 将读取配置文件（“.py”文件）并将其解析为类似字典的对象。

```plaintext
Open-Sora
└── configs                        -> Configs for training & inference
    ├── opensora                   -> STDiT related configs
    │   ├── inference
    │   │   ├── 16x256x256.py      -> Sample videos 16 frames 256x256
    │   │   ├── 16x512x512.py      -> Sample videos 16 frames 512x512
    │   │   └── 64x512x512.py      -> Sample videos 64 frames 512x512
    │   └── train
    │       ├── 16x256x256.py      -> Train on videos 16 frames 256x256
    │       ├── 16x256x256.py      -> Train on videos 16 frames 256x256
    │       └── 64x512x512.py      -> Train on videos 64 frames 512x512
    ├── dit                        -> DiT related configs
    │   ├── inference
    │   │   ├── 1x256x256-class.py -> Sample images with ckpts from DiT
    │   │   ├── 1x256x256.py       -> Sample images with clip condition
    │   │   └── 16x256x256.py      -> Sample videos
    │   └── train
    │       ├── 1x256x256.py       -> Train on images with clip condition
    │       └── 16x256x256.py      -> Train on videos
    ├── latte                      -> Latte related configs
    └── pixart                     -> PixArt related configs
```

## 推理配置演示

要更改推理设置，可以直接修改相应的配置文件。或者您可以传递参数来覆盖配置文件（[config_utils.py](/opensora/utils/config_utils.py)）。要更改采样提示，您应该修改传递给“--prompt_path”参数的“.txt”文件。

```plaintext
--prompt_path ./assets/texts/t2v_samples.txt  -> prompt_path
--ckpt-path ./path/to/your/ckpt.pth           -> model["from_pretrained"]
```

下面提供了每个字段的解释。

```python
# Define sampling size
num_frames = 64               # number of frames
fps = 24 // 2                 # frames per second (divided by 2 for frame_interval=2)
image_size = (512, 512)       # image size (height, width)

# Define model
model = dict(
    type="STDiT-XL/2",        # Select model type (STDiT-XL/2, DiT-XL/2, etc.)
    space_scale=1.0,          # (Optional) Space positional encoding scale (new height / old height)
    time_scale=2 / 3,         # (Optional) Time positional encoding scale (new frame_interval / old frame_interval)
    enable_flash_attn=True,    # (Optional) Speed up training and inference with flash attention
    enable_layernorm_kernel=True, # (Optional) Speed up training and inference with fused kernel
    from_pretrained="PRETRAINED_MODEL",  # (Optional) Load from pretrained model
    no_temporal_pos_emb=True,  # (Optional) Disable temporal positional encoding (for image)
)
vae = dict(
    type="VideoAutoencoderKL", # Select VAE type
    from_pretrained="stabilityai/sd-vae-ft-ema", # Load from pretrained VAE
    micro_batch_size=128,      # VAE with micro batch size to save memory
)
text_encoder = dict(
    type="t5",                 # Select text encoder type (t5, clip)
    from_pretrained="DeepFloyd/t5-v1_1-xxl", # Load from pretrained text encoder
    model_max_length=120,      # Maximum length of input text
)
scheduler = dict(
    type="iddpm",              # Select scheduler type (iddpm, dpm-solver)
    num_sampling_steps=100,    # Number of sampling steps
    cfg_scale=7.0,             # hyper-parameter for classifier-free diffusion
)
dtype = "fp16"                 # Computation type (fp16, fp32, bf16)

# Other settings
batch_size = 1                 # batch size
seed = 42                      # random seed
prompt_path = "./assets/texts/t2v_samples.txt"  # path to prompt file
save_dir = "./samples"         # path to save samples
```

## 训练配置演示

```python
# Define sampling size
num_frames = 64
frame_interval = 2             # sample every 2 frames
image_size = (512, 512)

# Define dataset
root = None                    # root path to the dataset
data_path = "CSV_PATH"         # path to the csv file
use_image_transform = False    # True if training on images
num_workers = 4                # number of workers for dataloader

# Define acceleration
dtype = "bf16"                 # Computation type (fp16, bf16)
grad_checkpoint = True         # Use gradient checkpointing
plugin = "zero2"               # Plugin for distributed training (zero2, zero2-seq)
sp_size = 1                    # Sequence parallelism size (1 for no sequence parallelism)

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=2 / 3,
    from_pretrained="YOUR_PRETRAINED_MODEL",
    enable_flash_attn=True,        # Enable flash attention
    enable_layernorm_kernel=True, # Enable layernorm kernel
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    micro_batch_size=128,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
    shardformer=True,           # Enable shardformer for T5 acceleration
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",      # Default 1000 timesteps
)

# Others
seed = 42
outputs = "outputs"             # path to save checkpoints
wandb = False                   # Use wandb for logging

epochs = 1000                   # number of epochs (just large enough, kill when satisfied)
log_every = 10
ckpt_every = 250
load = None                     # path to resume training

batch_size = 4
lr = 2e-5
grad_clip = 1.0                 # gradient clipping
```

# Repo & Config Structure

## Repo Structure

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

## Configs

Our config files follows [MMEgine](https://github.com/open-mmlab/mmengine). MMEngine will reads the config file (a `.py` file) and parse it into a dictionary-like object.

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

## Inference config demos

To change the inference settings, you can directly modify the corresponding config file. Or you can pass arguments to overwrite the config file ([config_utils.py](/opensora/utils/config_utils.py)). To change sampling prompts, you should modify the `.txt` file passed to the `--prompt_path` argument.

```plaintext
--prompt_path ./assets/texts/t2v_samples.txt  -> prompt_path
--ckpt-path ./path/to/your/ckpt.pth           -> model["from_pretrained"]
```

The explanation of each field is provided below.

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
    enable_flashattn=True,    # (Optional) Speed up training and inference with flash attention
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
    cfg_channel=3,             # how many channels to use for classifier-free diffusion, if None, use all channels
)
dtype = "fp16"                 # Computation type (fp16, fp32, bf16)

# Other settings
batch_size = 1                 # batch size
seed = 42                      # random seed
prompt_path = "./assets/texts/t2v_samples.txt"  # path to prompt file
save_dir = "./samples"         # path to save samples
```

## Training config demos

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
    enable_flashattn=True,        # Enable flash attention
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

## Bucket Configs

To enable dynamic training (for STDiT2), use `VariableVideoText` dataset, and set the `bucket_config` in the config. An example is:

```python
bucket_config = {
    "240p": {16: (1.0, 16), 32: (1.0, 8), 64: (1.0, 4), 128: (1.0, 2)},
    "256": {1: (1.0, 256)},
    "512": {1: (1.0, 80)},
    "480p": {1: (1.0, 52), 16: (0.5, 4), 32: (0.0, None)},
    "720p": {16: (1.0, 2), 32: (0.0, None)},
    "1024": {1: (1.0, 20)},
    "1080p": {1: (1.0, 8)},
}
```

This looks a bit difficult to understand at the first glance. Let's understand this config step by step.

### Why bucket?

Dynamic training needs to support training on different resolution (HxW), different aspect ratio (H/W), and different frame length. There are several possible ways to achieve this:

- NaViT: support dynamic size within the same batch by masking, without efficiency loss. However, the system is a bit complex to implement, and may not benefit from optimized kernels such as flash attention.
- FiT: support dynamic size within the same batch by padding. However, padding different resolutions to the same size is not efficient.
- PixArt: support dynamic size in different batches by bucketing, but the size must be the same within the same batch, and only a fixed number of size can be applied. With the same size in a batch, we do not need to implement complex masking or padding.

Bucketing means we pre-define some fixed resolution, and allocate different samples to different bucket. The concern for bucketing is listed below. But we can see that the concern is not a big issue in our case.

- The bucket size is limited to a fixed number: First, in real-world applications, only a few aspect ratios (9:16, 3:4) and resolutions (240p, 1080p) are commonly used. Second, we find trained models can generalize well to unseen resolutions.
- The size in each batch is the same, breaks the i.i.d. assumption: Since we are using multiple GPUs, the local batches on different GPUs have different sizes. We did not see a significant performance drop due to this issue.
- The may not be enough samples to fill each bucket and the distribution may be biased: First, our dataset is large enough to fill each bucket when local batch size is not too large. Second, we should analyze the data's distribution on sizes and define the bucket size accordingly. Third, an unbalanced distribution did not affect the training process significantly.
- Different resolutions and frame lengths may have different processing speed: Different from PixArt, which only deals with aspect ratios of similar resolutions (similar token numbers), we need to consider the processing speed of different resolutions and frame lengths. We can use the `bucket_config` to define the batch size for each bucket to ensure the processing speed is similar.

### Three-level bucket

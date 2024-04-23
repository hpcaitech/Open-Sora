
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

## Inference-long specific arguments

The [`inference-long.py`](/scripts/inference-long.py) script is used to generate long videos, and it also provides all functions of the [`inference.py`](/scripts/inference.py) script. The following arguments are specific to the `inference-long.py` script.

```python
loop = 10
condition_frame_length = 4
reference_path = ["one.png;two.mp4"]
mask_strategy = ["0,0,0,1,0;0,0,0,1,-1"]
```

To generate a long video of any time, our strategy is to generate a video with a fixed length first, and then use the last `condition_frame_length` number of frames for the next video generation. This will loop for `loop` times. Thus, the total length of the video is `loop * (num_frames - condition_frame_length) + condition_frame_length`.

To condition the generation on images or videos, we introduce the `mask_strategy`. It is 5 number tuples separated by `;`.  Each tuple indicate an insertion of the condition image or video to the target generation. The meaning of each number is:

- First number: the index of the condition image or video in the `reference_path`. (0 means one.png, and 1 means two.mp4)
- Second number: the loop index of the condition image or video. (0 means the first loop, 1 means the second loop, etc.)
- Third number: the start frame of the condition image or video. (0 means the first frame, and images only have one frame)
- Fourth number: the number of frames to insert. (1 means insert one frame, and images only have one frame)
- Fifth number: the location to insert. (0 means insert at the beginning, 1 means insert at the end, and -1 means insert at the end of the video)

Thus, "0,0,0,1,-1" means insert the first frame of one.png at the end of the video at the first loop.

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

### Three-level bucket

We design a three-level bucket: `(resolution, num_frames, aspect_ratios)`. The resolution and aspect ratios is predefined in [aspect.py](/opensora/datasets/aspect.py). Commonly used resolutions (e.g., 240p, 1080p) are supported, and the name represents the number of pixels (e.g., 240p is 240x426, however, we define 240p to represent any size with HxW approximately 240x426=102240 pixels). The aspect ratios are defined for each resolution. You do not need to define the aspect ratios in the `bucket_config`.

The `num_frames` is the number of frames in each sample, with `num_frames=1` especially for images. If `frame_intervals` is not 1, a bucket with `num_frames=k` will contain videos with `k*frame_intervals` frames except for images. Only a video with more than `num_frames` and more than `resolution` pixels will be likely to be put into the bucket.

The two number defined in the bucket config is `(keep_prob, batch_size)`. Since the memory and speed of samples from different buckets may be different, we use `batch_size` to balance the processing speed. Since our computation is limited, we cannot process videos with their original resolution as stated in OpenAI's sora's report. Thus, we give a `keep_prob` to control the number of samples in each bucket. The `keep_prob` is the probability to keep a sample in the bucket. Let's take the following config as an example:

```python
bucket_config = {
    "480p": {16: (1.0, 8),},
    "720p": {16: (0.5, 4),},
    "1080p": {16: (0.2, 2)},
    "4K", {16: (0.1, 1)},
}
```

Given a 2K video with more than 16 frames, the program will first try to put it into bucket "1080p" since it has a larger resolution than 1080p but less than 4K. Since the `keep_prob` for 1080p is 20%, a random number is generated, and if it is less than 0.2, the video will be put into the bucket. If the video is not put into the bucket, the program will try to put it into the "720p" bucket. Since the `keep_prob` for 720p is 50%, the video has a 50% chance to be put into the bucket. If the video is not put into the bucket, the program will try to put it into the "480p" bucket directly as it is the smallest resolution.

### Examples

Let's see some simple examples to understand the bucket config. First, the aspect ratio bucket is compulsory, if you want to modify this you need to add your own resolution definition in [aspect.py](/opensora/datasets/aspect.py). Then, to keep only 256x256 resolution and 16 frames as OpenSora 1.0, you can use the following config:

```python
bucket_config = {
    "256": {16: (1.0, 8)},
}
```

If you want to train a model supporting different resolutions of images, you can use the following config:

```python
bucket_config = {
    "256": {1: (1.0, 256)},
    "512": {1: (1.0, 80)},
    "480p": {1: (1.0, 52)},
    "1024": {1: (1.0, 20)},
    "1080p": {1: (1.0, 8)},
}
```

Or if you find the number of high-resolution images is too large, you can modify the `keep_prob` to reduce the number of samples in the bucket:

```python
bucket_config = {
    "256": {1: (1.0, 256)},
    "512": {1: (0.8, 80)},
    "480p": {1: (0.5, 52)},
    "1024": {1: (0.5, 20)},
    "1080p": {1: (0.2, 8)},
}
```

And similarly for videos:

```python
bucket_config = {
    "240p": {16: (1.0, 16), 32: (1.0, 8), 64: (1.0, 4), 128: (1.0, 2)},
    "480p": {16: (1.0, 4)},
    "720p": {16: (0.5, 2)},
}
```

Note that in the above case, a video with 480p resolution and more than 16 frames will all go into bucket `("480p", 16)`, since they all satisfy this bucket's requirement. But training long videos with 480p resolution may be slow, so you can modify the config as follows to enforce the video with more than 32 frames to go into the 240p bucket.

```python
bucket_config = {
    "240p": {16: (1.0, 16), 32: (1.0, 8), 64: (1.0, 4), 128: (1.0, 2)},
    "480p": {16: (1.0, 4), 32: (0.0, None)},
    "720p": {16: (0.5, 2)},
}
```

Combine the above examples together, we think you can understand the bucket config provided at the beginning of this section and in the config files.

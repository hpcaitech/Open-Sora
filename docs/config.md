# Config Guide

- [Inference Config](#inference-config)
- [Advanced Inference config](#advanced-inference-config)
- [Inference Args](#inference-args)
- [Training Config](#training-config)
- [Training Args](#training-args)
- [Training Bucket Configs](#training-bucket-configs)

Our config files follows [MMEgine](https://github.com/open-mmlab/mmengine). MMEngine will reads the config file (a `.py` file) and parse it into a dictionary-like object. We expose some fields in the config file to the command line arguments (defined in [opensora/utils/config_util.py](/opensora/utils/config_utils.py)). To change the inference settings, you can directly modify the corresponding config file. Or you can pass arguments to overwrite the config file.

## Inference Config

The explanation of each field is provided below.

```python
# Define sampling size
num_frames = 64               # number of frames, 1 means image
fps = 24                      # frames per second (condition for generation)
frame_interval = 3            # output video will have fps/frame_interval frames per second
image_size = (240, 426)       # image size (height, width)

# Define model
model = dict(
    type="STDiT2-XL/2",       # Select model type (STDiT-XL/2, DiT-XL/2, etc.)
    from_pretrained="PRETRAINED_MODEL",  # (Optional) Load from pretrained model
    input_sq_size=512,        # Base spatial position embedding size
    qk_norm=True,             # Normalize query and key in attention
    enable_flash_attn=True,    # (Optional) Speed up training and inference with flash attention
    # Turn enable_flash_attn to False if you skip flashattn installation
    enable_layernorm_kernel=True, # (Optional) Speed up training and inference with fused kernel
    # Turn enable_layernorm_kernel to False if you skip apex installation
)
vae = dict(
    type="VideoAutoencoderKL", # Select VAE type
    from_pretrained="stabilityai/sd-vae-ft-ema", # Load from pretrained VAE
    micro_batch_size=4,        # VAE with micro batch size to save memory
)
text_encoder = dict(
    type="t5",                 # Select text encoder type (t5, clip)
    from_pretrained="DeepFloyd/t5-v1_1-xxl", # Load from pretrained text encoder
    model_max_length=200,      # Maximum length of input text
)
scheduler = dict(
    type="iddpm",              # Select scheduler type (iddpm, dpm-solver)
    num_sampling_steps=100,    # Number of sampling steps
    cfg_scale=7.0,             # hyper-parameter for classifier-free diffusion
    cfg_channel=3,             # how many channels to use for classifier-free diffusion, if None, use all channels
)
dtype = "bf16"                 # Computation type (fp16, fp32, bf16)

# Condition
prompt_path = "./assets/texts/t2v_samples.txt" # path to prompt file
prompt = None                  # prompt has higher priority than prompt_path

# Other settings
batch_size = 1                 # batch size
seed = 42                      # random seed
save_dir = "./samples"         # path to save samples
```

## Advanced Inference config

The [`inference-long.py`](/scripts/inference-long.py) script is used to generate long videos, and it also provides all functions of the [`inference.py`](/scripts/inference.py) script. The following arguments are specific to the `inference-long.py` script.

```python
loop = 10
condition_frame_length = 4
reference_path = [
    "https://cdn.openai.com/tmp/s/interp/d0.mp4",
    None,
    "assets/images/condition/wave.png",
]
mask_strategy = [
    "0,0,0,0,8,0.3",
    None,
    "0,0,0,0,1;0,0,0,-1,1",
]
```

The following figure provides an illustration of the `mask_strategy`:

![mask_strategy](/assets/readme/report_mask_config.png)

To generate a long video of infinite time, our strategy is to generate a video with a fixed length first, and then use the last `condition_frame_length` number of frames for the next video generation. This will loop for `loop` times. Thus, the total length of the video is `loop * (num_frames - condition_frame_length) + condition_frame_length`.

To condition the generation on images or videos, we introduce the `mask_strategy`. It is 6 number tuples separated by `;`.  Each tuple indicate an insertion of the condition image or video to the target generation. The meaning of each number is:

- **First number**: the loop index of the condition image or video. (0 means the first loop, 1 means the second loop, etc.)
- **Second number**: the index of the condition image or video in the `reference_path`.
- **Third number**: the start frame of the condition image or video. (0 means the first frame, and images only have one frame)
- **Fourth number**: the location to insert. (0 means insert at the beginning, 1 means insert at the end, and -1 means insert at the end of the video)
- **Fifth number**: the number of frames to insert. (1 means insert one frame, and images only have one frame)
- **Sixth number**: the edit rate of the condition image or video. (0 means no edit, 1 means full edit).

To facilitate usage, we also accept passing the reference path and mask strategy as a json appended to the prompt. For example,

```plaintext
'Drone view of waves crashing against the rugged cliffs along Big Sur\'s garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff\'s edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff\'s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.{"reference_path": "assets/images/condition/cliff.png", "mask_strategy": "0"}'
```

## Inference Args

You can use `python scripts/inference.py --help` to see the following arguments:

- `--seed`: random seed
- `--ckpt-path`: path to the checkpoint (`model["from_pretrained"]`)
- `--batch-size`: batch size
- `--save-dir`: path to save samples
- `--sample-name`: if None, the sample will be name by `sample_{index}.mp4/png`, otherwise, the sample will be named by `{sample_name}_{index}.mp4/png`
- `--start-index`: start index of the sample
- `--end-index`: end index of the sample
- `--num-sample`: number of samples to generate for each prompt. The sample will be suffixed by `-0`, `-1`, `-2`, etc.
- `--prompt-as-path`: if True, use the prompt as the name for saving samples
- `--prompt-path`: path to the prompt file
- `--prompt`: prompt string list
- `--num-frames`: number of frames
- `--fps`: frames per second
- `--image-size`: image size
- `--num-sampling-steps`: number of sampling steps (`scheduler["num_sampling_steps"]`)
- `--cfg-scale`: hyper-parameter for classifier-free diffusion (`scheduler["cfg_scale"]`)
- `--loop`: loop for long video generation
- `--condition-frame-length`: condition frame length for long video generation
- `--reference-path`: reference path for long video generation
- `--mask-strategy`: mask strategy for long video generation

Example commands for inference can be found in [commands.md](/docs/commands.md).

## Training Config

```python
# Define dataset
dataset = dict(
    type="VariableVideoTextDataset",   # Select dataset type
    # VideoTextDataset for OpenSora 1.0, VariableVideoTextDataset for OpenSora 1.1 and 1.2
    data_path=None,                    # Path to the dataset
    num_frames=None,                   # Number of frames, set None since we support dynamic training
    frame_interval=3,                  # Frame interval
    image_size=(None, None),           # Image size, set None since we support dynamic training
    transform_name="resize_crop",      # Transform name
)
# bucket config usage see next section
bucket_config = {
    "144p": {1: (1.0, 48), 16: (1.0, 17), 32: (1.0, 9), 64: (1.0, 4), 128: (1.0, 1)},
    "256": {1: (0.8, 254), 16: (0.5, 17), 32: (0.5, 9), 64: (0.5, 4), 128: (0.5, 1)},
    "240p": {1: (0.1, 20), 16: (0.9, 17), 32: (0.8, 9), 64: (0.8, 4), 128: (0.8, 2)},
    "512": {1: (0.5, 86), 16: (0.2, 4), 32: (0.2, 2), 64: (0.2, 1), 128: (0.0, None)},
    "480p": {1: (0.4, 54), 16: (0.4, 4), 32: (0.0, None)},
    "720p": {1: (0.1, 20), 16: (0.1, 2), 32: (0.0, None)},
    "1024": {1: (0.3, 20)},
    "1080p": {1: (0.4, 8)},
}
# mask ratio in training
mask_ratios = {
    "identity": 0.75,                   # 75% no mask
    "quarter_random": 0.025,      # 2.5% random mask with 1 frame to 1/4 #frames
    "quarter_head": 0.025,        # 2.5% mask at the beginning with 1 frame to 1/4 #frames
    "quarter_tail": 0.025,        # 2.5% mask at the end with 1 frame to 1/4 #frames
    "quarter_head_tail": 0.05,    # 5% mask at the beginning and end with 1 frame to 1/4 #frames
    "image_random": 0.025,        # 2.5% random mask with 1 image to 1/4 #images
    "image_head": 0.025,          # 2.5% mask at the beginning with 1 image to 1/4 #images
    "image_tail": 0.025,          # 2.5% mask at the end with 1 image to 1/4 #images
    "image_head_tail": 0.05,      # 5% mask at the beginning and end with 1 image to 1/4 #images
}

# Define acceleration
num_workers = 8                        # Number of workers for dataloader
num_bucket_build_workers = 16          # Number of workers for bucket building
dtype = "bf16"                         # Computation type (fp16, fp32, bf16)
grad_checkpoint = True                 # Use gradient checkpointing
plugin = "zero2"                       # Plugin for training
sp_size = 1                            # Sequence parallel size

# Define model
model = dict(
    type="STDiT2-XL/2",                # Select model type (STDiT-XL/2, DiT-XL/2, etc.)
    from_pretrained=None,              # Load from pretrained model
    input_sq_size=512,                 # Base spatial position embedding size
    qk_norm=True,                      # Normalize query and key in attention
    enable_flash_attn=True,             # (Optional) Speed up training and inference with flash attention
    enable_layernorm_kernel=True,      # (Optional) Speed up training and inference with fused kernel
)
vae = dict(
    type="VideoAutoencoderKL",         # Select VAE type
    from_pretrained="stabilityai/sd-vae-ft-ema",
    micro_batch_size=4,                # VAE with micro batch size to save memory
    local_files_only=True,             # Load from local files only (first time should be false)
)
text_encoder = dict(
    type="t5",                         # Select text encoder type (t5, clip)
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=200,              # Maximum length of input text
    shardformer=True,                  # Use shardformer
    local_files_only=True,             # Load from local files only (first time should be false)
)
scheduler = dict(
    type="iddpm",                      # Select scheduler type (iddpm, iddpm-speed)
    timestep_respacing="",
)

# Others
seed = 42                              # random seed
outputs = "outputs"                    # path to save outputs
wandb = False                          # Use wandb or not

epochs = 1000                          # Number of epochs (set a large number and kill the process when you want to stop)
log_every = 10
ckpt_every = 500
load = None

batch_size = None
lr = 2e-5
grad_clip = 1.0
```

## Training Args

- `--seed`: random seed
- `--ckpt-path`: path to the checkpoint (`model["from_pretrained"]`)
- `--batch-size`: batch size
- `--wandb`: use wandb or not
- `--load`: path to the checkpoint to load
- `--data-path`: path to the dataset (`dataset["data_path"]`)

See [commands.md](/docs/commands.md) for example commands.

## Training Bucket Configs

We support multi-resolution/aspect-ratio/num_frames training with bucket. To enable dynamic training (for STDiT2), use `VariableVideoText` dataset, and set the `bucket_config` in the config. An example is:

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

![bucket](/assets/readme/report_bucket.png)

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

If you want to train a model supporting different resolutions of images, you can use the following config (example [image.py](/configs/opensora-v1-1/train/image.py)):

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

And similarly for videos (example [video.py](/configs/opensora-v1-1/train/video.py)):

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

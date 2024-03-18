# Commands

## Inference

You can modify corresponding config files to change the inference settings. See more details [here](/docs/structure.md#inference-config-demos).

### Inference with DiT pretrained on ImageNet

The following command automatically downloads the pretrained weights on ImageNet and runs inference.

```bash
python scripts/inference.py configs/dit/inference/1x256x256-class.py --ckpt-path DiT-XL-2-256x256.pt
```

### Inference with Latte pretrained on UCF101

The following command automatically downloads the pretrained weights on UCF101 and runs inference.

```bash
python scripts/inference.py configs/latte/inference/16x256x256-class.py --ckpt-path Latte-XL-2-256x256-ucf101.pt
```

### Inference with PixArt-α pretrained weights

Download T5 into `./pretrained_models` and run the following command.

```bash
# 256x256
torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/pixart/inference/1x256x256.py --ckpt-path PixArt-XL-2-256x256.pth

# 512x512
torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/pixart/inference/1x512x512.py --ckpt-path PixArt-XL-2-512x512.pth

# 1024 multi-scale
torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/pixart/inference/1x1024MS.py --ckpt-path PixArt-XL-2-1024MS.pth
```

### Inference with checkpoints saved during training

During training, an experiment logging folder is created in `outputs` directory. Under each checkpoint folder, e.g. `epoch12-global_step2000`, there is a `ema.pt` and the shared `model` folder. Run the following command to perform inference.

```bash
# inference with ema model
torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/opensora/inference/16x256x256.py --ckpt-path outputs/001-STDiT-XL-2/epoch12-global_step2000/ema.pt

# inference with model
torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/opensora/inference/16x256x256.py --ckpt-path outputs/001-STDiT-XL-2/epoch12-global_step2000

# inference with sequence parallelism
# sequence parallelism is enabled automatically when nproc_per_node is larger than 1
torchrun --standalone --nproc_per_node 2 scripts/inference.py configs/opensora/inference/16x256x256.py --ckpt-path outputs/001-STDiT-XL-2/epoch12-global_step2000
```

The second command will automatically generate a `model_ckpt.pt` file in the checkpoint folder.

### Inference Hyperparameters

1. DPM-solver is good at fast inference for images. However, the video result is not satisfactory. You can use it for fast demo purpose.

```python
type="dmp-solver"
num_sampling_steps=20
```

2. You can use [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)'s finetuned VAE decoder on videos for inference (consumes more memory). However, we do not see significant improvement in the video result. To use it, download [the pretrained weights](https://huggingface.co/maxin-cn/Latte/tree/main/t2v_required_models/vae_temporal_decoder) into `./pretrained_models/vae_temporal_decoder` and modify the config file as follows.

```python
vae = dict(
    type="VideoAutoencoderKLTemporalDecoder",
    from_pretrained="pretrained_models/vae_temporal_decoder",
)
```

## Training

To resume training, run the following command. ``--load`` different from ``--ckpt-path`` as it loads the optimizer and dataloader states.

```bash
torchrun --nnodes=1 --nproc_per_node=8 scripts/train.py configs/opensora/train/64x512x512.py --data-path YOUR_CSV_PATH --load YOUR_PRETRAINED_CKPT
```

To enable wandb logging, add `--wandb` to the command.

```bash
WANDB_API_KEY=YOUR_WANDB_API_KEY torchrun --nnodes=1 --nproc_per_node=8 scripts/train.py configs/opensora/train/64x512x512.py --data-path YOUR_CSV_PATH --wandb True
```

You can modify corresponding config files to change the training settings. See more details [here](/docs/structure.md#training-config-demos).

### Training Hyperparameters

1. `dtype` is the data type for training. Only `fp16` and `bf16` are supported. ColossalAI automatically enables the mixed precision training for `fp16` and `bf16`. During training, we find `bf16` more stable.

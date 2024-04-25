# Commands

- [Inference](#inference)
  - [Inference with Open-Sora 1.1](#inference-with-open-sora-11)
  - [Inference with DiT pretrained on ImageNet](#inference-with-dit-pretrained-on-imagenet)
  - [Inference with Latte pretrained on UCF101](#inference-with-latte-pretrained-on-ucf101)
  - [Inference with PixArt-α pretrained weights](#inference-with-pixart-α-pretrained-weights)
  - [Inference with checkpoints saved during training](#inference-with-checkpoints-saved-during-training)
  - [Inference Hyperparameters](#inference-hyperparameters)
- [Training](#training)
  - [Training Hyperparameters](#training-hyperparameters)
- [Search batch size for buckets](#search-batch-size-for-buckets)

## Inference

You can modify corresponding config files to change the inference settings. See more details [here](/docs/structure.md#inference-config-demos).

### Inference with Open-Sora 1.1

Since Open-Sora 1.1 supports inference with dynamic input size, you can pass the input size as an argument.

```bash
# image sampling with prompt path
python scripts/inference.py configs/opensora-v1-1/inference/sample.py \
    --ckpt-path CKPT_PATH --prompt-path assets/texts/t2i_samples.txt --num-frames 1 --image-size 1024 1024

# image sampling with prompt
python scripts/inference.py configs/opensora-v1-1/inference/sample.py \
    --ckpt-path CKPT_PATH --prompt "A beautiful sunset over the city" --num-frames 1 --image-size 1024 1024

# video sampling
python scripts/inference.py configs/opensora-v1-1/inference/sample.py \
    --ckpt-path CKPT_PATH --prompt "A beautiful sunset over the city" --num-frames 16 --image-size 480 854
```

You can adjust the `--num-frames` and `--image-size` to generate different results. We recommend you to use the same image size as the training resolution, which is defined in [aspect.py](/opensora/datasets/aspect.py). Some examples are shown below.

- 240p
  - 16:9 240x426
  - 3:4 276x368
  - 1:1 320x320
- 480p
  - 16:9 480x854
  - 3:4 554x738
  - 1:1 640x640
- 720p
  - 16:9 720x1280
  - 3:4 832x1110
  - 1:1 960x960

`inference-long.py` is compatible with `inference.py` and supports advanced features.

```bash
# image condition
python scripts/inference-long.py configs/opensora-v1-1/inference/sample.py --ckpt-path CKPT_PATH \
  --num-frames 32 --image-size 240 426 --sample-name image-cond \
  --prompt 'A breathtaking sunrise scene.{"reference_path": "assets/images/condition/wave.png","mask_strategy": "0"}'

# video extending
python scripts/inference-long.py configs/opensora-v1-1/inference/sample.py --ckpt-path CKPT_PATH \
  --num-frames 32 --image-size 240 426 --sample-name image-cond \
  --prompt 'A car driving on the ocean.{"reference_path": "https://cdn.openai.com/tmp/s/interp/d0.mp4","mask_strategy": "0,0,0,-8,8"}'

# long video generation
python scripts/inference-long.py configs/opensora-v1-1/inference/sample.py --ckpt-path CKPT_PATH \
  --num-frames 32 --image-size 240 426 --loop 16 --condition-frame-length 8 --sample-name long \
  --prompt '|0|a white jeep equipped with a roof rack driving on a dirt road in a coniferous forest.|2|a white jeep equipped with a roof rack driving on a dirt road in the desert.|4|a white jeep equipped with a roof rack driving on a dirt road in a mountain.|6|A white jeep equipped with a roof rack driving on a dirt road in a city.|8|a white jeep equipped with a roof rack driving on a dirt road on the surface of a river.|10|a white jeep equipped with a roof rack driving on a dirt road under the lake.|12|a white jeep equipped with a roof rack flying into the sky.|14|a white jeep equipped with a roof rack driving in the universe. Earth is the background.{"reference_path": "https://cdn.openai.com/tmp/s/interp/d0.mp4", "mask_strategy": "0,0,0,0,16"}'

# video connecting
python scripts/inference-long.py configs/opensora-v1-1/inference/sample.py --ckpt-path CKPT_PATH \
  --num-frames 32 --image-size 240 426 --sample-name connect \
  --prompt 'A breathtaking sunrise scene.{"reference_path": "assets/images/condition/sunset1.png;assets/images/condition/sunset2.png","mask_strategy": "0;0,1,0,-1,1"}'

# video editing
python scripts/inference-long.py configs/opensora-v1-1/inference/sample.py --ckpt-path CKPT_PATH \
  --num-frames 32 --image-size 480 853 --sample-name edit \
  --prompt 'A cyberpunk-style city at night.{"reference_path": "https://cdn.pixabay.com/video/2021/10/12/91744-636709154_large.mp4","mask_strategy": "0,0,0,0,32,0.4"}'
```

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

## Search batch size for buckets

To search the batch size for buckets, run the following command.

```bash
torchrun --standalone --nproc_per_node 1 scripts/search_bs.py configs/opensora-v1-1/train/benchmark.py --data-path YOUR_CSV_PATH -o YOUR_OUTPUT_CONFIG_PATH --base-resolution 240p --base-frames 128 --batch-size-start 2 --batch-size-end 256 --batch-size-step 2
```

If your dataset is extremely large, you extract a subset of the dataset for the search.

```bash
# each bucket contains 1000 samples
python tools/datasets/split.py YOUR_CSV_PATH -o YOUR_SUBSET_CSV_PATH -c configs/opensora-v1-1/train/video.py -l 1000
```

If you want to control the batch size search more granularly, you can configure batch size start, end, and step in the config file.

Bucket config format:

1. `{ resolution: {num_frames: (prob, batch_size)} }`, in this case batch_size is ignored when searching
2. `{ resolution: {num_frames: (prob, (max_batch_size, ))} }`, batch_size is searched in the range `[batch_size_start, max_batch_size)`, batch_size_start is configured via CLI
3. `{ resolution: {num_frames: (prob, (min_batch_size, max_batch_size))} }`, batch_size is searched in the range `[min_batch_size, max_batch_size)`
4. `{ resolution: {num_frames: (prob, (min_batch_size, max_batch_size, step_size))} }`, batch_size is searched in the range `[min_batch_size, max_batch_size)` with step_size (grid search)
5. `{ resolution: {num_frames: (0.0, None)} }`, this bucket will not be used

Here is an example of the bucket config:

```python
bucket_config = {

    "240p": {
        16: (1.0, (2, 32)),
        32: (1.0, (2, 16)),
        64: (1.0, (2, 8)),
        128: (1.0, (2, 6)),
    },
    "256": {1: (1.0, (128, 300))},
    "512": {1: (0.5, (64, 128))},
    "480p": {1: (0.4, (32, 128)), 16: (0.4, (2, 32)), 32: (0.0, None)},
    "720p": {16: (0.1, (2, 16)), 32: (0.0, None)},  # No examples now
    "1024": {1: (0.3, (8, 64))},
    "1080p": {1: (0.3, (2, 32))},
}
```

It will print the best batch size (and corresponding step time) for each bucket and save the output config file.

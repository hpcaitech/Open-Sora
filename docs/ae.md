# Step by step to train and evaluate an video autoencoder

xxx (4x32x32) hy (4x8x8). briefly introduce benefits.

## Data Preparation

Follow this [guide](./train.md#prepare-dataset) to prepare the __DATASET__ for training and inference. You may use our provided dataset or custom ones.

To use custom dataset, pass the argument `--dataset.data_path <your_data_path>` to the following training or inference command.

## Training

We propose a video autoencoder architecture based on [DCAE](https://github.com/mit-han-lab/efficientvit), the __Video DC-AE__, and train it from scratch on 8xGPUs for 3 weeks.

We first train with the following command:

```bash
torchrun --nproc_per_node 8 scripts/vae/train.py configs/vae/train/video_dc_ae.py --wandb True
```

When the model is almost converged, we add a discriminator and continue to train the model with the checkpoint `model_ckpt` using the following command:

```bash
torchrun --nproc_per_node 8 scripts/vae/train.py configs/vae/train/video_dc_ae_disc.py --model.from_pretrained <model_ckpt> --wandb True
```

## Inference

### Video DC-AE

Use the following code to reconstruct the videos using our trained `Video DC-AE`:

```bash
torchrun --nproc_per_node 1 --standalone scripts/vae/inference.py configs/vae/inference/video_dc_ae.py --save-dir samples/dcae
```

### Hunyuan Video

Alternatively, we have incorporated [HunyuanVideo vae](https://github.com/Tencent/HunyuanVideo) into our code, you may run inference with the following command:

```bash
torchrun --nproc_per_node 1 --standalone scripts/vae/inference.py configs/vae/inference/hunyuanvideo_vae.py --save-dir samples/hunyuanvideo_vae
```

## Config Interpretation

All VAE configs are located in `configs/vae/`, divided into configs for training (`configs/vae/train`) and for inference (`configs/vae/inference`).

### Training Config

For training, the same config rules as [those](./train.md#config) for the diffusion model are applied.

Loss config and each loss meaning.

### Inference Config

For VAE inference, we have replicated the tiling mechanism in hunyuan to our Video DC-AE, which can be turned on with the following:

```bash
model = dict(
    ...,
    use_spatial_tiling=True,
    use_temporal_tiling=True,
    spatial_tile_size=256,
    temporal_tile_size=32,
    tile_overlap_factor=0.25,
    ...,
)
```

By default, both spatial tiling and temporal tiling are turned on for the best performance.
Since our Video DC-AE is trained on 256px videos of 32 frames only, `spatial_tile_size` should be set to 256 and `temporal_tile_size` should be set to 32.
If you train your own Video DC-AE with other resolutions and length, you may adjust the values accordingly.

You can specify the directory to store output samples with `--save_dir <your_dir>` or setting it in config, for instance:

```bash
save_dir = "./samples"
```

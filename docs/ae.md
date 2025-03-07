# Step by step to train and evaluate an video autoencoder (AE)
Inspired by [SANA](https://arxiv.org/abs/2410.10629), we aim to drastically increase the compression ratio in the AE. We propose a video autoencoder architecture based on [DC-AE](https://github.com/mit-han-lab/efficientvit), the __Video DC-AE__, which compression the video by 4x in the temporal dimension and 32x32 in the spatial dimension. Compared to [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)'s VAE of 4x8x8, our proposed AE has a much higher spatial compression ratio.
Thus, we can effectively reduce the token length in the diffusion model by a total of 16x (assuming the same patch sizes), drastically increase both training and inference speed.

## Data Preparation

Follow this [guide](./train.md#prepare-dataset) to prepare the __DATASET__ for training and inference. You may use our provided dataset or custom ones.

To use custom dataset, pass the argument `--dataset.data_path <your_data_path>` to the following training or inference command.

## Training

We train our __Video DC-AE__ from scratch on 8xGPUs for 3 weeks.

We first train with the following command:

```bash
torchrun --nproc_per_node 8 scripts/vae/train.py configs/vae/train/video_dc_ae.py
```

When the model is almost converged, we add a discriminator and continue to train the model with the checkpoint `model_ckpt` using the following command:

```bash
torchrun --nproc_per_node 8 scripts/vae/train.py configs/vae/train/video_dc_ae_disc.py --model.from_pretrained <model_ckpt>
```
You may pass the flag `--wandb True` if you have a [wandb](https://wandb.ai/home) account and wish to track the training progress online.

## Inference

Download the relevant weights following [this guide](../README.md#model-download). Alternatively, you may use your own trained model by passing the following flag `--model.from_pretrained <your_model_ckpt_path>`.

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

All AE configs are located in `configs/vae/`, divided into configs for training (`configs/vae/train`) and for inference (`configs/vae/inference`).

### Training Config

For training, the same config rules as [those](./train.md#config) for the diffusion model are applied.

<details>
<summary> <b>Loss Config</b> </summary>
Our __Video DC-AE__ is based on the [DC-AE](https://github.com/mit-han-lab/efficientvit) architecture, which doesn't have a variational component. Thus, our training simply composes of the *reconstruction loss* and the *perceptual loss*.
Experimentally, we found that setting a ratio of 0.5 for the perceptual loss is effective.

```python
vae_loss_config = dict(
    perceptual_loss_weight=0.5, # weigh the perceptual loss by 0.5
    kl_loss_weight=0,           # no KL loss
)
```

In a later stage, we include a discriminator, and the training loss for the ae has an additional generator loss component, where we use a small ratio of 0.05 to weigh the loss calculated:
```python
gen_loss_config = dict(
    gen_start=0,                # include generator loss from step 0 onwards          
    disc_weight=0.05,           # weigh the loss by 0.05
)
```

The discriminator we use is trained from scratch, and it's loss is simply the hinged loss:
```python
disc_loss_config = dict(
    disc_start=0,               # update the discriminator from step 0 onwards
    disc_loss_type="hinge",     # the discriminator loss type
)
```
</details>

<details>
<summary> <b> Data Bucket Config </b> </summary>
For the data bucket, we used 32 frames of 256px videos to train our AE.
```python
bucket_config = {
    "256px_ar1:1": {32: (1.0, 1)},
}
```
</details>

<details>
<summary> <b>Train with more frames or higher resolutions</b> </summary>

If you train with longer frames or larger resolutions, you may increase the `spatial_tile_size` and `temporal_tile_size` during inference without degrading the AE performance (see [Inference Config](ae.md#inference-config)). This may give you advantage of faster AE inference such as when training the diffusion model (although at the cost of slower AE training). 

You may increase the video frames to 96 (although multiples of 4 works, we generally recommend to use frame numbers of multiples of 32):

```python
bucket_config = {
    "256px_ar1:1": {96: (1.0, 1)},
}
grad_checkpoint = True
```
or train for higher resolution such as 512px:
```python
bucket_config = {
    "512px_ar1:1": {32: (1.0, 1)},
}
grad_checkpoint = True
```
Note that gradient checkpoint needs to be turned on in order to avoid prevent OOM error.

Moreover, if `grad_checkpointing` is set to `True` in discriminator training, you need to pass the flag `--model.disc_off_grad_ckpt True` or simply set in the config:
```python
grad_checkpoint = True
model = dict(
    disc_off_grad_ckpt = True, # set to true if your `grad_checkpoint` is True
)
```
This is to make sure the discriminator loss will have a gradient at the laster later during adaptive loss calculation.
</details>




### Inference Config

For AE inference, we have replicated the tiling mechanism in hunyuan to our Video DC-AE, which can be turned on with the following:

```python
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

```python
save_dir = "./samples"
```

# Commands

## Train VAE with 8 machines

```bash
colossalai run --hostfile hostfile --nproc_per_node 8 scripts/train_opensoravae_v1_3.py configs/vae_v1_3/train/video_16z.py --data-path YOUR_CSV_PATH --ckpt-path YOUR_PRETRAINED_CKPT --wandb True > logs/train_opensoravae_v1_3.log 2>&1 &
```

## Evaluate VAE performance

* If ``VID_PATH`` is not specified, use the default uses vid100 used in `eval/loss`;
* We use image1k used in `eval/loss` for image evaluation;
* eval stats are saved to `${CKPT_PATH}/eval/ folder`.

```bash
VID_PATH=/home/shenchenhui/data/eval_loss/forest_vid_100/pixabay_forest_vid_100.csv CUDA_VISIBLE_DEVICES=0 bash eval/vae/launch.sh pretrained_models/OpenSoraVAE_V1_3/model.pt
```

## Inference

We can set optimization options in vae config:
```bash
vae = dict(
    type="OpenSoraVAE_V1_3",
    from_pretrained=None,
    z_channels=16,
    shift=[...],
    scale=[...],
    micro_batch_size=1, # DON'T set during training of vae
    micro_batch_size_2d=4, # DON'T set during training of vae
    micro_frame_size=17, # DON'T set during training of vae
    use_tiled_conv3d=True,
    tile_size=4,
)
```

### Inference with VAE
set force-huggingface to True if loading the original pretrained huggingface model `pretrained_models/OpenSoraVAE_V1_3`.

```bash
# video
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/inference_opensoravae_v1_3.py configs/vae_v1_3/inference/video_16z_512x512.py  --data-path YOUR_CSV_PATH --save-dir ./samples/vae_16z/videos --ckpt-path YOUR_PRETRAINED_CKPT --force-huggingface False


# image
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/inference_opensoravae_v1_3.py configs/vae_v1_3/inference/image_16z.py  --data-path YOUR_CSV_PATH --save-dir ./samples/vae_16z/images/ --ckpt-path YOUR_PRETRAINED_CKPT --force-huggingface False
```

### Train DiT with freezed VAE

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 scripts/train.py configs/opensora-v1-3/train/stage1.py --data-path /mnt/ddn/sora/meta/pro_1_0_ddn/internvid_first_quarter_ext.csv
```

### Inference DiT with VAE

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/inference.py configs/opensora-v1-3/train/stage1.py --ckpt-path /mnt/ddn/sora/checkpoints/outputs/0245-STDiT3-XL-2/epoch0-global_step13000 --prompt-path assets/texts/t2v_samples.txt --save-dir samples/debug --num-frames 51 --resolution 360p --aspect-ratio 9:16 --sample-name sample_2s_360p --batch-size 1
```

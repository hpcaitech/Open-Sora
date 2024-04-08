# 命令

## 推理

您可以修改相应的配置文件来更改推理设置。在 [此处](/docs/structure.md#inference-config-demos) 查看更多详细信息。

### 在 ImageNet 上使用 DiT 预训练进行推理

以下命令会自动在 ImageNet 上下载预训练权重并运行推理。

```bash
python scripts/inference.py configs/dit/inference/1x256x256-class.py --ckpt-path DiT-XL-2-256x256.pt
```

### 在 UCF101 上使用 Latte 预训练进行推理

以下命令会自动下载 UCF101 上的预训练权重并运行推理。

```bash
python scripts/inference.py configs/latte/inference/16x256x256-class.py --ckpt-path Latte-XL-2-256x256-ucf101.pt
```

### 使用 PixArt-α 预训练权重进行推理

将 T5 下载到 `./pretrained_models` 并运行以下命令。

```bash
# 256x256
torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/pixart/inference/1x256x256.py --ckpt-path PixArt-XL-2-256x256.pth

# 512x512
torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/pixart/inference/1x512x512.py --ckpt-path PixArt-XL-2-512x512.pth

# 1024 multi-scale
torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/pixart/inference/1x1024MS.py --ckpt-path PixArt-XL-2-1024MS.pth
```

### 使用训练期间保存的 checkpoints 进行推理

在训练期间，会在 `outputs` 目录中创建一个实验日志记录文件夹。在每个 checkpoint 文件夹下（例如 `epoch12-global_step2000`），有一个 `ema.pt` 文件和共享的 `model` 文件夹。执行以下命令进行推理。

```bash
# 使用 ema 模型进行推理
torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/opensora/inference/16x256x256.py --ckpt-path outputs/001-STDiT-XL-2/epoch12-global_step2000/ema.pt

# 使用模型进行推理
torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/opensora/inference/16x256x256.py --ckpt-path outputs/001-STDiT-XL-2/epoch12-global_step2000

# 使用序列并行进行推理
# 当 nproc_per_node 大于 1 时，将自动启用序列并行
torchrun --standalone --nproc_per_node 2 scripts/inference.py configs/opensora/inference/16x256x256.py --ckpt-path outputs/001-STDiT-XL-2/epoch12-global_step2000
```

第二个命令将在 checkpoint 文件夹中自动生成一个 `model_ckpt.pt` 文件。

### 推理超参数

1. DPM 求解器擅长对图像进行快速推理。但是，它的视频推理的效果并不令人满意。若出于快速演示目的您可以使用这个求解器。

```python
type="dmp-solver"
num_sampling_steps=20
```

2. 您可以在视频推理上使用 [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) 微调的 VAE 解码器（消耗更多内存）。但是，我们没有看到视频推理效果有明显改善。要使用它，请将 [预训练权重](https://huggingface.co/maxin-cn/Latte/tree/main/t2v_required_models/vae_temporal_decoder) 下载到 `./pretrained_models/vae_temporal_decoder` 中，并修改配置文件，如下所示。

```python
vae = dict(
    type="VideoAutoencoderKLTemporalDecoder",
    from_pretrained="pretrained_models/vae_temporal_decoder",
)
```

## 训练

如果您要继续训练，请运行以下命令。参数 ``--load`` 和 ``--ckpt-path`` 不同之处在于，它会加载优化器和数据加载器的状态。

```bash
torchrun --nnodes=1 --nproc_per_node=8 scripts/train.py configs/opensora/train/64x512x512.py --data-path YOUR_CSV_PATH --load YOUR_PRETRAINED_CKPT
```

如果要启用 wandb 日志，请添加到 `--wandb` 参数到命令中。

```bash
WANDB_API_KEY=YOUR_WANDB_API_KEY torchrun --nnodes=1 --nproc_per_node=8 scripts/train.py configs/opensora/train/64x512x512.py --data-path YOUR_CSV_PATH --wandb True
```

您可以修改相应的配置文件来更改训练设置。在 [此处](/docs/structure.md#training-config-demos) 查看更多详细信息。

### 训练超参数

1. `dtype` 是用于训练的数据类型。仅支持 `fp16` 和 `bf16`。ColossalAI 自动启用 `fp16` 和 `bf16` 的混合精度训练。在训练过程中，我们发现 `bf16` 更稳定。

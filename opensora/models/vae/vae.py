import os

import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from einops import rearrange
from transformers import PretrainedConfig, PreTrainedModel

from opensora.registry import MODELS, build_module
from opensora.utils.ckpt_utils import load_checkpoint


@MODELS.register_module()
class VideoAutoencoderKL(nn.Module):
    def __init__(
        self,
        from_pretrained=None,
        micro_batch_size=None,
        cache_dir=None,
        local_files_only=False,
        subfolder=None,
        scaling_factor=0.18215,
    ):
        super().__init__()
        self.module = AutoencoderKL.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
        self.out_channels = self.module.config.latent_channels
        self.patch_size = (1, 8, 8)
        self.micro_batch_size = micro_batch_size
        self.scaling_factor = scaling_factor

    def encode(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")

        if self.micro_batch_size is None:
            x = self.module.encode(x).latent_dist.sample().mul_(self.scaling_factor)
        else:
            # NOTE: cannot be used for training
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.encode(x_bs).latent_dist.sample().mul_(self.scaling_factor)
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def decode(self, x, **kwargs):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        if self.micro_batch_size is None:
            x = self.module.decode(x / self.scaling_factor).sample
        else:
            # NOTE: cannot be used for training
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.decode(x_bs / self.scaling_factor).sample
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            # assert (
            #     input_size[i] is None or input_size[i] % self.patch_size[i] == 0
            # ), "Input size must be divisible by patch size"
            latent_size.append(input_size[i] // self.patch_size[i] if input_size[i] is not None else None)
        return latent_size

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


@MODELS.register_module()
class VideoAutoencoderKLTemporalDecoder(nn.Module):
    def __init__(self, from_pretrained=None, cache_dir=None, local_files_only=False):
        super().__init__()
        self.module = AutoencoderKLTemporalDecoder.from_pretrained(
            from_pretrained, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.out_channels = self.module.config.latent_channels
        self.patch_size = (1, 8, 8)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, x, **kwargs):
        B, _, T = x.shape[:3]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        x = self.module.decode(x / 0.18215, num_frames=T).sample
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            # assert (
            #     input_size[i] is None or input_size[i] % self.patch_size[i] == 0
            # ), "Input size must be divisible by patch size"
            latent_size.append(input_size[i] // self.patch_size[i] if input_size[i] is not None else None)
        return latent_size

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class VideoAutoencoderPipelineConfig(PretrainedConfig):
    model_type = "VideoAutoencoderPipeline"

    def __init__(
        self,
        vae_2d=None,
        vae_temporal=None,
        from_pretrained=None,
        freeze_vae_2d=False,
        cal_loss=False,
        micro_frame_size=None,
        shift=0.0,
        scale=1.0,
        **kwargs,
    ):
        self.vae_2d = vae_2d
        self.vae_temporal = vae_temporal
        self.from_pretrained = from_pretrained
        self.freeze_vae_2d = freeze_vae_2d
        self.cal_loss = cal_loss
        self.micro_frame_size = micro_frame_size
        self.shift = shift
        self.scale = scale
        super().__init__(**kwargs)


class VideoAutoencoderPipeline(PreTrainedModel):
    config_class = VideoAutoencoderPipelineConfig

    def __init__(self, config: VideoAutoencoderPipelineConfig):
        super().__init__(config=config)
        self.spatial_vae = build_module(config.vae_2d, MODELS)
        self.temporal_vae = build_module(config.vae_temporal, MODELS)
        self.cal_loss = config.cal_loss
        self.micro_frame_size = config.micro_frame_size
        self.micro_z_frame_size = self.temporal_vae.get_latent_size([config.micro_frame_size, None, None])[0]

        if config.freeze_vae_2d:
            for param in self.spatial_vae.parameters():
                param.requires_grad = False

        self.out_channels = self.temporal_vae.out_channels

        # normalization parameters
        scale = torch.tensor(config.scale)
        shift = torch.tensor(config.shift)
        if len(scale.shape) > 0:
            scale = scale[None, :, None, None, None]
        if len(shift.shape) > 0:
            shift = shift[None, :, None, None, None]
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    def encode(self, x):
        x_z = self.spatial_vae.encode(x)

        if self.micro_frame_size is None:
            posterior = self.temporal_vae.encode(x_z)
            z = posterior.sample()
        else:
            z_list = []
            for i in range(0, x_z.shape[2], self.micro_frame_size):
                x_z_bs = x_z[:, :, i : i + self.micro_frame_size]
                posterior = self.temporal_vae.encode(x_z_bs)
                z_list.append(posterior.sample())
            z = torch.cat(z_list, dim=2)

        if self.cal_loss:
            return z, posterior, x_z
        else:
            return (z - self.shift) / self.scale

    def decode(self, z, num_frames=None):
        if not self.cal_loss:
            z = z * self.scale.to(z.dtype) + self.shift.to(z.dtype)

        if self.micro_frame_size is None:
            x_z = self.temporal_vae.decode(z, num_frames=num_frames)
            x = self.spatial_vae.decode(x_z)
        else:
            x_z_list = []
            for i in range(0, z.size(2), self.micro_z_frame_size):
                z_bs = z[:, :, i : i + self.micro_z_frame_size]
                x_z_bs = self.temporal_vae.decode(z_bs, num_frames=min(self.micro_frame_size, num_frames))
                x_z_list.append(x_z_bs)
                num_frames -= self.micro_frame_size
            x_z = torch.cat(x_z_list, dim=2)
            x = self.spatial_vae.decode(x_z)

        if self.cal_loss:
            return x, x_z
        else:
            return x

    def forward(self, x):
        assert self.cal_loss, "This method is only available when cal_loss is True"
        z, posterior, x_z = self.encode(x)
        x_rec, x_z_rec = self.decode(z, num_frames=x_z.shape[2])
        return x_rec, x_z_rec, z, posterior, x_z

    def get_latent_size(self, input_size):
        if self.micro_frame_size is None or input_size[0] is None:
            return self.temporal_vae.get_latent_size(self.spatial_vae.get_latent_size(input_size))
        else:
            sub_input_size = [self.micro_frame_size, input_size[1], input_size[2]]
            sub_latent_size = self.temporal_vae.get_latent_size(self.spatial_vae.get_latent_size(sub_input_size))
            sub_latent_size[0] = sub_latent_size[0] * (input_size[0] // self.micro_frame_size)
            remain_temporal_size = [input_size[0] % self.micro_frame_size, None, None]
            if remain_temporal_size[0] > 0:
                remain_size = self.temporal_vae.get_latent_size(remain_temporal_size)
                sub_latent_size[0] += remain_size[0]
            return sub_latent_size

    def get_temporal_last_layer(self):
        return self.temporal_vae.decoder.conv_out.conv.weight

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


@MODELS.register_module()
def OpenSoraVAE_V1_2(
    micro_batch_size=4,
    micro_frame_size=17,
    from_pretrained=None,
    local_files_only=False,
    freeze_vae_2d=False,
    cal_loss=False,
    force_huggingface=False,
):
    vae_2d = dict(
        type="VideoAutoencoderKL",
        from_pretrained="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        subfolder="vae",
        micro_batch_size=micro_batch_size,
        local_files_only=local_files_only,
    )
    vae_temporal = dict(
        type="VAE_Temporal_SD",
        from_pretrained=None,
    )
    shift = (-0.10, 0.34, 0.27, 0.98)
    scale = (3.85, 2.32, 2.33, 3.06)
    kwargs = dict(
        vae_2d=vae_2d,
        vae_temporal=vae_temporal,
        freeze_vae_2d=freeze_vae_2d,
        cal_loss=cal_loss,
        micro_frame_size=micro_frame_size,
        shift=shift,
        scale=scale,
    )

    if force_huggingface or (from_pretrained is not None and not os.path.exists(from_pretrained)):
        model = VideoAutoencoderPipeline.from_pretrained(from_pretrained, **kwargs)
    else:
        config = VideoAutoencoderPipelineConfig(**kwargs)
        model = VideoAutoencoderPipeline(config)

        if from_pretrained:
            load_checkpoint(model, from_pretrained)
    return model

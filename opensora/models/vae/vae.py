import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from einops import rearrange

from opensora.registry import MODELS, build_module
from opensora.utils.ckpt_utils import load_checkpoint


@MODELS.register_module()
class VideoAutoencoderKL(nn.Module):
    def __init__(
        self, from_pretrained=None, micro_batch_size=None, cache_dir=None, local_files_only=False, subfolder=None
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

    def encode(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")

        if self.micro_batch_size is None:
            x = self.module.encode(x).latent_dist.sample().mul_(0.18215)
        else:
            # NOTE: cannot be used for training
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.encode(x_bs).latent_dist.sample().mul_(0.18215)
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def decode(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        if self.micro_batch_size is None:
            x = self.module.decode(x / 0.18215).sample
        else:
            # NOTE: cannot be used for training
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.decode(x_bs / 0.18215).sample
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

    def decode(self, x):
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


@MODELS.register_module()
class VideoAutoencoderPipeline(nn.Module):
    def __init__(self, vae_2d=None, vae_temporal=None, freeze_vae_2d=False, from_pretrained=None):
        super().__init__()
        self.spatial_vae = build_module(vae_2d, MODELS)
        self.temporal_vae = build_module(vae_temporal, MODELS)

        if from_pretrained is not None:
            load_checkpoint(self, from_pretrained)
        if freeze_vae_2d:
            for param in self.spatial_vae.parameters():
                param.requires_grad = False

    def encode(self, x, training=True):
        x_z = self.spatial_vae.encode(x)
        posterior = self.temporal_vae.encode(x_z)
        z = posterior.sample()
        if training:
            return z, posterior, x_z
        return z

    def decode(self, z, num_frames=None, training=True):
        x_z = self.temporal_vae.decode(z, num_frames=num_frames)
        x = self.spatial_vae.decode(x_z)
        if training:
            return x, x_z
        return x

    def forward(self, x):
        z, posterior, x_z = self.encode(x, training=True)
        x_rec, x_z_rec = self.decode(z, num_frames=x_z.shape[2], training=True)
        return x_rec, x_z_rec, z, posterior, x_z

    def get_latent_size(self, input_size):
        return self.temporal_vae.get_latent_size(self.spatial_vae.get_latent_size(input_size))

    def get_temporal_last_layer(self):
        return self.temporal_vae.decoder.conv_out.conv.weight
    
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

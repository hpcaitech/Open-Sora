import torch.nn as nn
from einops import rearrange


class AutoencoderKLWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.module = vae
        self.out_channels = vae.config.latent_channels
        self.patch_size = [1, 8, 8]
        self.eval()

    def encode(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.module.encode(x).latent_dist.sample().mul_(0.18215)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=B)
        return x

    def decode(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.module.decode(x / 0.18215).sample
        x = rearrange(x, "(b t) c h w -> b c t h w", b=B)
        return x

    def get_latent_size(self, input_size):
        for i in range(3):
            assert input_size[i] % self.patch_size[i] == 0, "Input size must be divisible by patch size"
        input_size = [input_size[i] // self.patch_size[i] for i in range(3)]
        return input_size


class AutoencoderKLTemporalDecoderWrapper(AutoencoderKLWrapper):
    def encode(self, x):
        raise NotImplementedError

    def decode(self, x):
        B = x.shape[0]
        T = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.module.decode(x / 0.18215, num_frames=T).sample
        x = rearrange(x, "(b t) c h w -> b c t h w", b=B)
        return x


def load_vae(name):
    if "stabilityai/sd-vae-ft-" in name:
        from diffusers.models import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(name)
        return AutoencoderKLWrapper(vae)
    elif name == "vae_temporal_decoder":
        from diffusers.models import AutoencoderKLTemporalDecoder

        vae = AutoencoderKLTemporalDecoder.from_pretrained("pretrained_models/vae_temporal_decoder")
        return AutoencoderKLTemporalDecoderWrapper(vae)
    else:
        raise ValueError(f"Unknown VAE name: {name}")

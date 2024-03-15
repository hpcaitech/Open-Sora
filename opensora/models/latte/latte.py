# Adapted from Latte
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Latte:  https://github.com/Vchitect/Latte
# DiT:    https://github.com/facebookresearch/DiT/tree/main
# --------------------------------------------------------

import torch
from einops import rearrange, repeat

from opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora.models.dit import DiT
from opensora.registry import MODELS
from opensora.utils.ckpt_utils import load_checkpoint


@MODELS.register_module()
class Latte(DiT):
    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (B, C, T, H, W) tensor of inputs
        t: (B,) tensor of diffusion timesteps
        y: list of text
        """
        # origin inputs should be float32, cast to specified dtype
        x = x.to(self.dtype)

        # embedding
        x = self.x_embedder(x)  # (B, N, D)
        x = rearrange(x, "b (t s) d -> b t s d", t=self.num_temporal, s=self.num_spatial)
        x = x + self.pos_embed_spatial
        x = rearrange(x, "b t s d -> b (t s) d")

        t = self.t_embedder(t, dtype=x.dtype)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        if self.use_text_encoder:
            y = y.squeeze(1).squeeze(1)
        condition = t + y
        condition_spatial = repeat(condition, "b d -> (b t) d", t=self.num_temporal)
        condition_temporal = repeat(condition, "b d -> (b s) d", s=self.num_spatial)

        # blocks
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                # spatial
                x = rearrange(x, "b (t s) d -> (b t) s d", t=self.num_temporal, s=self.num_spatial)
                c = condition_spatial
            else:
                # temporal
                x = rearrange(x, "b (t s) d -> (b s) t d", t=self.num_temporal, s=self.num_spatial)
                c = condition_temporal
                if i == 1:
                    x = x + self.pos_embed_temporal

            x = auto_grad_checkpoint(block, x, c)  # (B, N, D)

            if i % 2 == 0:
                x = rearrange(x, "(b t) s d -> b (t s) d", t=self.num_temporal, s=self.num_spatial)
            else:
                x = rearrange(x, "(b s) t d -> b (t s) d", t=self.num_temporal, s=self.num_spatial)

        # final process
        x = self.final_layer(x, condition)  # (B, N, num_patches * out_channels)
        x = self.unpatchify(x)  # (B, out_channels, T, H, W)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x


@MODELS.register_module("Latte-XL/2")
def Latte_XL_2(from_pretrained=None, **kwargs):
    model = Latte(
        depth=28,
        hidden_size=1152,
        patch_size=(1, 2, 2),
        num_heads=16,
        **kwargs,
    )
    if from_pretrained is not None:
        load_checkpoint(model, from_pretrained)
    return model


@MODELS.register_module("Latte-XL/2x2")
def Latte_XL_2x2(from_pretrained=None, **kwargs):
    model = Latte(
        depth=28,
        hidden_size=1152,
        patch_size=(2, 2, 2),
        num_heads=16,
        **kwargs,
    )
    if from_pretrained is not None:
        load_checkpoint(model, from_pretrained)
    return model

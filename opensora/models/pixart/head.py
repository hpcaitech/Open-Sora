import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora.models.layers.blocks import (
    CaptionEmbedder,
    PatchEmbed3D,
    PositionEmbedding2D,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
)
from opensora.models.pixart.pixart import PixArtBlock
from opensora.registry import MODELS
from opensora.utils.ckpt_utils import load_checkpoint


@MODELS.register_module()
class PixArtHead(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=(None, None, None),
        input_sq_size=512,
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path: float = 0.0,
        caption_channels=4096,
        model_max_length=120,
        enable_flash_attn=True,
        enable_layernorm_kernel=True,
        enable_sequence_parallelism=False,
    ):
        super().__init__()
        assert enable_sequence_parallelism is False, "Sequence parallelism is not supported in this version."
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels

        # model size related
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # computation related
        self.drop_path = drop_path
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel

        # input size related
        self.patch_size = patch_size
        self.input_sq_size = input_sq_size
        self.pos_embed = PositionEmbedding2D(hidden_size)

        # embedding
        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )

        # spatial blocks
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                PixArtBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flash_attn=enable_flash_attn,
                    enable_layernorm_kernel=enable_layernorm_kernel,
                )
                for i in range(depth)
            ]
        )

        # final layer
        self.final_layer = T2IFinalLayer(hidden_size, np.prod(self.patch_size), self.out_channels)

    @property
    def dtype(self):
        return self.x_embedder.proj.weight.dtype

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def get_base_size(self, S):
        base_sizes = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
        value = S**0.5
        base_size = base_sizes[(base_sizes - value).abs().argmin()]
        return base_size

    def get_scale(self, height, width):
        scales = [0.5, 1, 2, 4, 8]
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.input_sq_size
        scale = scales[(torch.tensor(scales) - scale).abs().argmin()]
        return scale

    def encode_text(self, y, mask=None):
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, self.hidden_size)
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, self.hidden_size)
        return y, y_lens

    def forward(self, x, timestep, y, mask=None, height=None, width=None, cond=None, **kwargs):
        x, timestep, y = x.to(self.dtype), timestep.to(self.dtype), y.to(self.dtype)

        # === get pos embed ===
        _, _, Tx, Hx, Wx = x.size()
        assert Tx == 1, "T must be 1"
        T, H, W = self.get_dynamic_size(x)
        S = H * W
        base_size = self.get_base_size(S)
        scale = self.get_scale(height, width)
        pos_embed = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=x.dtype)  # (N, D)
        t0 = self.t_block(t)

        # === get y embed ===
        if y.size(0) != x.size(0):
            t_mult = x.size(0) // y.size(0)
            y = y.repeat_interleave(t_mult, dim=0)
            mask = mask.repeat_interleave(t_mult, dim=0)
        y, y_lens = self.encode_text(y, mask)

        # === get x embed ===
        x = self.x_embedder(x)  # (B, N, D)
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_embed
        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        # === add condition ===
        if cond is not None:
            x = x + cond

        # === blocks ===
        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, y, t0, y_lens)

        # final process
        x = self.final_layer(x, t, T=T, S=S)
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x


@MODELS.register_module("PixArt-HEAD-XL/2")
def PixArtHead_XL_2(from_pretrained=None, **kwargs):
    model = PixArtHead(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
    if from_pretrained is not None:
        load_checkpoint(model, from_pretrained)
    return model

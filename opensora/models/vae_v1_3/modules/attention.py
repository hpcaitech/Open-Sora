# modified from
# https://github.com/bornfly-detachment/asymmetric_magvitv2/blob/main/models/modules/attention.py


import torch
import torch.nn as nn
from einops import rearrange

from .conv import CausalConv3dPlainAR
from .ops import Normalize


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = CausalConv3dPlainAR(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = CausalConv3dPlainAR(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = CausalConv3dPlainAR(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = CausalConv3dPlainAR(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def _enable_tiled_conv3d(self, tile_size=16, tiled_dim=None, num_tiles=None):
        self.q._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)
        self.k._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)
        self.v._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)
        self.proj_out._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)

    def attention(self, h_) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        # print('h_', h_.shape)

        b, c, t, h, w = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c t h w -> b 1 (t h w) c").contiguous(), (q, k, v))
        h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return rearrange(h_, "b 1 (t h w) c -> b c t h w", t=t, h=h, w=w, c=c, b=b)

    def forward(self, x):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_

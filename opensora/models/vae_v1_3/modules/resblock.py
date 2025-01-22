# modified from
# https://github.com/bornfly-detachment/asymmetric_magvitv2/blob/main/models/modules/resblock.py


import logging

import torch
import torch.nn as nn

logpy = logging.getLogger(__name__)
from .conv import CausalConv3dPlainAR
from .ops import Normalize, nonlinearity


class Resnet3DBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, empty_cache=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = CausalConv3dPlainAR(in_channels, out_channels, kernel_size=3)

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CausalConv3dPlainAR(out_channels, out_channels, kernel_size=3)
        self.empty_cache = empty_cache
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConv3dPlainAR(in_channels, out_channels, kernel_size=3)
            else:
                self.nin_shortcut = CausalConv3dPlainAR(in_channels, out_channels, kernel_size=1)

    def _enable_tiled_conv3d(self, tile_size=16, tiled_dim=None, num_tiles=None):
        self.conv1._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)
        self.conv2._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)
        if hasattr(self, "conv_shortcut"):
            self.conv_shortcut._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)
        if hasattr(self, "nin_shortcut"):
            self.nin_shortcut._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)

    def forward(self, x, is_training=False):
        h = x.clone()
        h = self.norm1(h)

        if self.empty_cache and not is_training:
            torch.cuda.empty_cache()

        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        if is_training:
            x = x + h
        else:
            x.add_(h)
        return x

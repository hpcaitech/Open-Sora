# modified from
# https://github.com/bornfly-detachment/asymmetric_magvitv2/blob/main/models/modules/conv.py

import logging
from typing import Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

from opensora.models.vae_v1_3.utils import cast_tuple, is_odd
from opensora.models.layers.tiled_conv3d import TiledConv3d

logpy = logging.getLogger(__name__)


class CausalConv3dPlainAR(Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        pad_mode="constant",
        **kwargs,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = cast_tuple(kernel_size, 3)
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        self.time_kernel_size = time_kernel_size
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)
        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)
        self.pad_mode = pad_mode
        if isinstance(stride, int):
            stride = (stride, 1, 1)
        time_pad = dilation * (time_kernel_size - 1) + max((1 - stride[0]), 0)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        self.time_uncausal_padding = (width_pad, width_pad, height_pad, height_pad, 0, 0)
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def _enable_tiled_conv3d(self, tile_size=16, tiled_dim=None, num_tiles=None):
        # NOTE: currently not enable for stride != 1
        if self.conv.stride == (1, 1, 1):
            self.conv = TiledConv3d.from_native_conv3d(
                self.conv,
                num_tiles=num_tiles,
                tile_size=tile_size,
                tiled_dim=tiled_dim,
            )

    def forward(self, x):
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else "constant"
        x = F.pad(x, self.time_causal_padding, mode=pad_mode)

        x = self.conv(x)
        return x

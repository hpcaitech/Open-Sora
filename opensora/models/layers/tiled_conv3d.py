import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _size_3_t


class TiledConv3d(nn.Conv3d):
    """
    This is a PyTorch module for 3D Convolution with tiling.

    Args:
        num_tiles (int, optional): The number of tiles to partition the input tensor. If not given, the input tensor will be partitioned into tiles of size 16.
        tiled_dim (int, optional): The dimension to be tiled. If not given, the dimension with the largest size will be chosen.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        num_tiles=None,  # if have num_tiles, use num_tiles, else use tile_size to determine
        tile_size=16,
        tiled_dim=None,
        exclude_temporal_dim=False,
    ):
        super(TiledConv3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.num_tiles = num_tiles
        self.tile_size = tile_size
        self.exclude_temporal_dim = exclude_temporal_dim

        self.tiled_dim = tiled_dim
        if self.tiled_dim is not None:
            assert self.tiled_dim in [2, 3, 4], "tiled_dim should be 2, 3, or 4"

    def forward(self, x):
        assert self.padding_mode == "zeros", "Only support zero padding"

        # 1. determinte the tiled dim and num_tiles
        if self.tiled_dim is None:
            # if not given, choose the dim with th largest dimension
            B, C, T, H, W = x.size()
            if not self.exclude_temporal_dim:
                tiled_dim = np.argmax([T, H, W]).item() + 2
            else:
                tiled_dim = np.argmax([H, W]).item() + 3
        else:
            tiled_dim = self.tiled_dim

        if self.num_tiles is None:
            # if num_tiles is not given, tile the dim into 16
            num_tiles = x.size(tiled_dim) // self.tile_size
        else:
            num_tiles = self.num_tiles

        assert num_tiles < x.size(tiled_dim), "num_tiles should be less than the size of the tiled dimension"

        # we compute the number of conv ops in total
        # and compute the number of tiles
        # we then allocate the tiles based on the number of conv ops
        tiled_dim_size = x.size(tiled_dim)
        kernel_size = self.kernel_size[tiled_dim - 2]
        num_slibing_elements = (self.kernel_size[tiled_dim - 2] - 1) // 2
        stride = self.stride[tiled_dim - 2]
        padding = self.padding[tiled_dim - 2]
        dilation = self.dilation[tiled_dim - 2]
        assert dilation == 1, "Only support dilation=1"
        num_conv_ops = math.floor((tiled_dim_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

        # distribute the conv ops into tiles
        conv_ops_per_partition = num_conv_ops // num_tiles
        remainder = num_conv_ops % num_tiles
        num_conv_ops_per_tile = [conv_ops_per_partition] * num_tiles
        for i in range(remainder):
            num_conv_ops_per_tile[i] += 1

        # 2. compute with tiling
        out_list = []
        conv_ops_processed = 0

        # we apply padding ahead of computation to make the tiling sharding easier
        padding_3d = [0, 0, 0, 0, 0, 0]
        padding_on_tiled_dim = self.padding[tiled_dim - 2]
        if padding_on_tiled_dim > 0:
            padding_3d[2 * (tiled_dim - 2)] = padding_on_tiled_dim
            padding_3d[2 * (tiled_dim - 2) + 1] = padding_on_tiled_dim
        x = F.pad(x, padding_3d, "constant", 0)
        padding_list = list(self.padding)
        padding_list[tiled_dim - 2] = 0

        for i in range(num_tiles):
            # comput the center point where convolution is applied
            start_idx = num_slibing_elements + stride * conv_ops_processed
            end_idx = start_idx + stride * num_conv_ops_per_tile[i]
            conv_ops_processed += num_conv_ops_per_tile[i]

            # the actual boundary requires considering the silbling elements
            start_idx -= num_slibing_elements
            end_idx += num_slibing_elements

            cur_partition = x.narrow(tiled_dim, start_idx, end_idx - start_idx)
            out_partition = F.conv3d(
                cur_partition, self.weight, self.bias, self.stride, padding_list, self.dilation, self.groups
            )
            out_list.append(out_partition)

        merged_out = torch.cat(out_list, dim=tiled_dim)

        return merged_out

    @staticmethod
    def from_native_conv3d(module: nn.Conv3d, num_tiles=None, tile_size=16, tiled_dim=None, exclude_temporal_dim=False):
        """
        This is a static method for easy conversion from the PyTorch native Conv3d module to the TiledConv3d module.

        Examples:
            tiled_conv3d = TiledConv3d.from_native_conv3d(conv3d, num_tiles=32, tiled_dim=3)
        """
        auto_tiled_conv3d = TiledConv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            device=module.weight.device,
            dtype=module.weight.dtype,
            num_tiles=num_tiles,
            tiled_dim=tiled_dim,
            tile_size=tile_size,
            exclude_temporal_dim=exclude_temporal_dim,
        )
        auto_tiled_conv3d.weight = module.weight
        auto_tiled_conv3d.bias = module.bias
        return auto_tiled_conv3d

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, num_tiles={self.num_tiles}, tile_size={self.tile_size}, tiled_dim={self.tiled_dim}, exclude_temporal_dim={self.exclude_temporal_dim})"

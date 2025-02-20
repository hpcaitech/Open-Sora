# modified from
# https://github.com/bornfly-detachment/asymmetric_magvitv2/blob/main/models/vae/encoder.py


import torch.nn as nn

from opensora.registry import MODELS

from .modules.attention import AttnBlock
from .modules.conv import CausalConv3dPlainAR
from .modules.ops import Normalize, nonlinearity
from .modules.resblock import Resnet3DBlock
from .modules.updownsample import Downsample2D, Downsample3D
from .utils import is_odd


@MODELS.register_module()
class VideoEncoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        z_channels,
        double_z=True,
        down_sampling_layer=[1, 2],
        micro_batch_size_2d=None,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.down_sampling_layer = down_sampling_layer

        # downsampling
        self.conv_in = CausalConv3dPlainAR(in_channels, self.ch, kernel_size=3)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    Resnet3DBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                if i_level in self.down_sampling_layer:
                    down.downsample = Downsample3D(block_in, resamp_with_conv, stride=(2, 2, 2))
                else:
                    down.downsample = Downsample2D(block_in, resamp_with_conv, micro_batch_size_2d=micro_batch_size_2d)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = Resnet3DBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = Resnet3DBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = CausalConv3dPlainAR(block_in, 2 * z_channels if double_z else z_channels, kernel_size=3)

    def _enable_tiled_conv3d(self, tile_size=16, tiled_dim=None, num_tiles=None):
        if hasattr(self.conv_in, "_enable_tiled_conv3d"):
            self.conv_in._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                if hasattr(self.down[i_level].block[i_block], "_enable_tiled_conv3d"):
                    self.down[i_level].block[i_block]._enable_tiled_conv3d(
                        tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles
                    )
            if i_level != self.num_resolutions - 1:
                if hasattr(self.down[i_level].downsample, "_enable_tiled_conv3d"):
                    self.down[i_level].downsample._enable_tiled_conv3d(
                        tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles
                    )

        if hasattr(self.mid.block_1, "_enable_tiled_conv3d"):
            self.mid.block_1._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)
        if hasattr(self.mid.attn_1, "_enable_tiled_conv3d"):
            self.mid.attn_1._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)
        if hasattr(self.mid.block_2, "_enable_tiled_conv3d"):
            self.mid.block_2._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)

        if hasattr(self.conv_out, "_enable_tiled_conv3d"):
            self.conv_out._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)

    def forward(self, x, is_training=False):
        # timestep embedding
        t = x.size(2)
        if is_odd(t):
            temporal_length = max(t // (2 ** len(self.down_sampling_layer)) + 1, 1)
        else:
            temporal_length = t // (2 ** len(self.down_sampling_layer))

        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, is_training=is_training)

            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h, is_training=is_training)

        h = self.mid.block_1(h, is_training=is_training)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, is_training=is_training)

        h = self.norm_out(h)
        h = nonlinearity(h, is_training=is_training)
        h = self.conv_out(h)

        if h.shape[2] != temporal_length:
            print("shape strange: ", h.shape, x.shape)
        return h

# modified from
# https://github.com/bornfly-detachment/asymmetric_magvitv2/blob/main/models/vae/decoder.py


import torch.nn as nn
import torch.nn.functional as F

from opensora.models.layers.tiled_conv3d import TiledConv3d
from opensora.registry import MODELS

from .modules.attention import AttnBlock
from .modules.conv import CausalConv3dPlainAR
from .modules.ops import Normalize, nonlinearity
from .modules.resblock import Resnet3DBlock
from .modules.updownsample import Upsample2D, Upsample3D


@MODELS.register_module()
class VideoDecoder(nn.Module):
    def __init__(
        self,
        ch,
        out_ch,
        in_channels,
        z_channels,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        dropout=0.0,
        resamp_with_conv=True,
        temporal_up_layers=[2, 3],
        temporal_downsample=4,
        micro_batch_size_2d=None,
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.temporal_up_layers = temporal_up_layers
        self.temporal_downsample = temporal_downsample

        (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = CausalConv3dPlainAR(z_channels, block_in, kernel_size=3)
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

        # upsampling
        self.up_id = len(self.temporal_up_layers)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    Resnet3DBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                if i_level in self.temporal_up_layers:
                    up.upsample = Upsample3D(block_in, resamp_with_conv)
                else:
                    up.upsample = Upsample2D(block_in, resamp_with_conv, micro_batch_size_2d=micro_batch_size_2d)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.time_causal_padding = (1, 1, 1, 1, 2, 0)
        self.time_uncausal_padding = (1, 1, 1, 1, 0, 0)
        self.conv_out = nn.Conv3d(block_in, out_ch, kernel_size=3)

    def get_last_layer(self):
        return self.conv_out.weight

    def _enable_tiled_conv3d(self, tile_size=16, tiled_dim=None, num_tiles=None):
        if hasattr(self.conv_in, "_enable_tiled_conv3d"):
            self.conv_in._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)

        if hasattr(self.mid.block_1, "_enable_tiled_conv3d"):
            self.mid.block_1._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)
        if hasattr(self.mid.attn_1, "_enable_tiled_conv3d"):
            self.mid.attn_1._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)
        if hasattr(self.mid.block_2, "_enable_tiled_conv3d"):
            self.mid.block_2._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                if hasattr(self.up[i_level].block[i_block], "_enable_tiled_conv3d"):
                    self.up[i_level].block[i_block]._enable_tiled_conv3d(
                        tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles
                    )
            if i_level != 0:
                if hasattr(self.up[i_level].upsample, "_enable_tiled_conv3d"):
                    self.up[i_level].upsample._enable_tiled_conv3d(
                        tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles
                    )

        self.conv_out = TiledConv3d.from_native_conv3d(
            self.conv_out,
            num_tiles=num_tiles,
            tile_size=tile_size,
            tiled_dim=tiled_dim,
        )

    def forward(self, z, is_training=False):
        self.last_z_shape = z.shape

        h = self.conv_in(z)
        h = self.mid.block_1(h, is_training=is_training)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, is_training=is_training)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, is_training=is_training)
            if i_level != 0:
                h = self.up[i_level].upsample(h, is_split=True, is_training=is_training)

        h = self.norm_out(h)
        h = nonlinearity(h, is_training=is_training)
        h = F.pad(h, self.time_causal_padding, mode="constant")
        h = self.conv_out(h)
        h = h[:, :, (self.temporal_downsample - 1) :]

        return h

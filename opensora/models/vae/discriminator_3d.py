"""3D StyleGAN discriminator."""

import functools
import math
from typing import Any

import torch
import torch.nn as nn

# TODO: torch.nn.init.xavier_uniform_
# default_kernel_init = nn.initializers.xavier_uniform()

class ResBlock(nn.Module):
    """3D StyleGAN ResBlock for D."""

    def __init__(
        self,
        in_channels,
        filters,
        activation_fn,
        num_groups=32,
        device="cpu",
        dtype=torch.bfloat16,
    ):
        super().__init__()

        self.filters = filters
        self.activation_fn = activation_fn

        # SCH: NOTE: although paper says conv (X->Y, Y->Y), original code implementation is (X->X, X->Y), we follow code
        self.conv1 = nn.Conv3d(in_channels, in_channels, (3,3,3)) # TODO: need to init to xavier_uniform
        self.norm1 = nn.GroupNorm(num_groups, in_channels, device=device, dtype=dtype)
        self.avg_pool_with_t = nn.AvgPool3d((2,2,2),count_include_pad=False)
        self.conv2 = nn.Conv3d(in_channels, self.filters,(1,1,1), use_bias=False) # need to init to xavier_uniform
        self.conv3 = nn.Conv3d(in_channels, self.filters, (3,3,3)) # need to init to xavier_uniform
        self.norm2 = nn.GroupNorm(num_groups, self.filters, device=device, dtype=dtype)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation_fn(x)

        x = self.avg_pool_with_t(x)
        residual = self.avg_pool_with_t(residual)
        residual = self.conv2(residual)
        x = self.conv3(x)
        x = self.norm2(x)
        x = self.activation_fn(x)
        out = (residual + x) / math.sqrt(2)
        return out
    
class StyleGANDiscriminator(nn.Module):
    """StyleGAN Discriminator."""
    def __init__(
        self,
        config,
        image_size,
        num_frames,
        discriminator_in_channels = 3,
        discriminator_filters = 64,
        discriminator_channel_multipliers = (2,4,4,4,4),
        num_groups=32,
        dtype = torch.bfloat16,
        device="cpu",
    ):
        self.config = config
        self.dtype = dtype
        self.input_size = image_size
        self.filters = discriminator_filters
        self.activation_fn = nn.LeakyReLu(negative_slope=0.2)
        self.channel_multipliers = discriminator_channel_multipliers

        self.conv1 = nn.Conv3d(discriminator_in_channels, self.filters, (3, 3, 3)) # need to init to xavier_uniform
        
        prev_filters = self.filters # record in_channels
        self.num_blocks = len(self.channel_multipliers)
        self.res_block_list = []
        for i in range(self.num_blocks):
            filters = self.filters * self.channel_multipliers[i]
            self.res_block_list.append(ResBlock(prev_filters, filters, self.activation_fn))
            prev_filters = filters # update in_channels 

        self.conv2 = nn.Conv3d(prev_filters, prev_filters, (3,3,3)) # need to init to xavier_uniform

        self.norm1 = nn.GroupNorm(num_groups, prev_filters, dtype=dtype, device=device)

        # TODO: what is the in_features
        scale_factor = 2 ** len(self.num_blocks)
        time_scaled = num_frames / scale_factor
        image_scaled = image_size / scale_factor
        in_features = prev_filters * time_scaled * image_scaled * image_scaled  # (C*T*W*H)
        self.linear1 = nn.Linear(in_features, prev_filters, device=device, dtype=dtype) # need to init to xavier_uniform
        self.linear2 = nn.Linear(prev_filters, 1, device=device, dtype=dtype) # need to init to xavier_uniform

    def forward(self, x):

        x = self.conv1(x)
        x = self.activation_fn(x)
        
        for i in range(self.num_blocks):
            x = self.res_block_list[i](x)

        x = self.conv2(x)
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = x.reshape((x.shape[0], -1)) # SCH: [B, (C * T * W * H)] ? 

        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x

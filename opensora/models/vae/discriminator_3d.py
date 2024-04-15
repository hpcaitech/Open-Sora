"""3D StyleGAN discriminator."""

import functools
import math
from typing import Any

import torch
import torch.nn as nn

# TODO: torch.nn.init.xavier_uniform_
# default_kernel_init = nn.initializers.xavier_uniform()

def xavier_uniform_weight_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        # print("initialized module to xavier_uniform:", m)

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
        self.conv1 = nn.Conv3d(in_channels, in_channels, (3,3,3), padding=1, dtype=dtype, device=device) # TODO: need to init to xavier_uniform
        self.norm1 = nn.GroupNorm(num_groups, in_channels, device=device, dtype=dtype)
        self.avg_pool_with_t = nn.AvgPool3d((2,2,2),count_include_pad=False)
        self.conv2 = nn.Conv3d(in_channels, self.filters,(1,1,1), bias=False, dtype=dtype, device=device) # need to init to xavier_uniform
        self.conv3 = nn.Conv3d(in_channels, self.filters, (3,3,3), padding=1, dtype=dtype, device=device) # need to init to xavier_uniform
        self.norm2 = nn.GroupNorm(num_groups, self.filters, device=device, dtype=dtype)
        self.apply(xavier_uniform_weight_init)

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
        image_size = (128, 128),
        num_frames = 17,
        in_channels = 3,
        filters = 128,
        channel_multipliers = (2,4,4,4,4),
        num_groups=32,
        dtype = torch.bfloat16,
        device="cpu",
    ):
        super().__init__()
        self.dtype = dtype
        self.input_size = image_size
        self.filters = filters
        self.activation_fn = nn.LeakyReLU(negative_slope=0.2)
        self.channel_multipliers = channel_multipliers

        self.conv1 = nn.Conv3d(in_channels, self.filters, (3, 3, 3), padding=1, dtype=dtype, device=device) # need to init to xavier_uniform
        
        prev_filters = self.filters # record in_channels
        self.num_blocks = len(self.channel_multipliers)
        self.res_block_list = []
        for i in range(self.num_blocks):
            filters = self.filters * self.channel_multipliers[i]
            self.res_block_list.append(ResBlock(prev_filters, filters, self.activation_fn, dtype=dtype,device=device))
            prev_filters = filters # update in_channels 

        self.conv2 = nn.Conv3d(prev_filters, prev_filters, (3,3,3), padding=1, dtype=dtype, device=device) # need to init to xavier_uniform

        self.norm1 = nn.GroupNorm(num_groups, prev_filters, dtype=dtype, device=device)

        scale_factor = 2 ** self.num_blocks
        if num_frames % scale_factor != 0: # SCH: NOTE: has first frame which would be padded before usage
            time_scaled = num_frames // scale_factor + 1
        else:
            time_scaled = num_frames / scale_factor
        assert self.input_size[0] % scale_factor == 0, f"image width {self.input_size[0]} is not divisible by scale factor {scale_factor}"
        assert self.input_size[1] % scale_factor == 0, f"image height {self.input_size[1]} is not divisible by scale factor {scale_factor}"

        w_scaled, h_scaled = self.input_size[0] / scale_factor, self.input_size[1] / scale_factor
        in_features = int(prev_filters * time_scaled * w_scaled * h_scaled)  # (C*T*W*H)
        self.linear1 = nn.Linear(in_features, prev_filters, device=device, dtype=dtype) # need to init to xavier_uniform
        self.linear2 = nn.Linear(prev_filters, 1, device=device, dtype=dtype) # need to init to xavier_uniform

        self.apply(xavier_uniform_weight_init)

    def forward(self, x):

        x = self.conv1(x)
        x = self.activation_fn(x)
        
        for i in range(self.num_blocks):
            x = self.res_block_list[i](x)

        x = self.conv2(x)
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = x.reshape((x.shape[0], -1)) # SCH: [B, (C * T * W * H)] ? 
        breakpoint()
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x

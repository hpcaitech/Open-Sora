import functools
from typing import Any, Dict, Tuple, Type, Union, Sequence, Optional
from absl import logging
import ml_collections
import torch
import torch.nn as nn 
import numpy as np 
from numpy import typing as nptyping
from opensora.models.vae import model_utils 
from opensora.registry import MODELS

"""Encoder and Decoder stuctures with 3D CNNs."""

"""
NOTE:
    removed LayerNorm since not used in this arch
    GroupNorm: flax uses default `epsilon=1e-06`, whereas torch uses `eps=1e-05`
    for average pool and upsample, input shape needs to be [N,C,T,H,W] --> if not, adjust the scale factors accordingly

    !!! opensora read video into [B,C,T,H,W] format output

TODO:
    check data dimensions format
"""

class ResBlock(nn.Module):
    def __init__(
            self, 
            in_out_channels, # SCH: added
            filters, 
            # norm_fn, # SCH: removed, use GN
            conv_fn, 
            dtype="fp16", 
            activation_fn=nn.ReLU, 
            use_conv_shortcut=False,
            num_groups=32,
    ):
        super().__init__()
        self.filters = filters
        
        # SCH: MAGVIT uses GroupNorm by default
        self.norm1 = nn.GroupNorm(num_groups, in_out_channels, dtype=dtype)
        self.conv1 = conv_fn(in_out_channels, self.filters, kernel_size=(3, 3, 3), use_bias=False)
        self.norm2 = nn.GroupNorm(num_groups, self.filters, dtype=dtype)
        self.conv2 = conv_fn(self.filters, self.filters, kernel_size=(3, 3, 3), use_bias=False)
        if self.use_conv_shortcut:
            self.conv3 = conv_fn(self.filters, self.filters, kernel_size=(3, 3, 3), use_bias=False)
        else:
            self.conv3 = conv_fn(self.filters, self.filters, kernel_size=(1, 1, 1), use_bias=False)

        self.dtype = dtype
        self.activate = activation_fn()
        self.use_conv_shortcut = use_conv_shortcut

    def forward(self, x):
        input_dim = x.shape[-1]
        residual = x
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activate(x)
        x = self.conv2(x) 
        if input_dim != self.filters:
            residual = self.conv3(residual)
        return x + residual 
    
def _get_selected_flags(total_len: int, select_len: int, suffix: bool):
    assert select_len <= total_len
    selected = np.zeros(total_len, dtype=bool)
    if not suffix:
        selected[:select_len] = True
    else:
        selected[-select_len:] = True
    return selected


class Encoder(nn.Module):
    """Encoder Blocks."""
    def __init__(self, 
        filters = 64,
        num_res_blocks = 3,
        channel_multipliers = (1, 2, 2, 4),
        temporal_downsample = (True, True, False),
        num_groups = 32, # for nn.GroupNorm
        in_out_channels = 3, # SCH: added, in_channels at the start
        latent_embed_dim = 256, 
        conv_downsample = False, 
        custom_conv_padding = None,
        activation_fn = 'swish',
        dtype="bf16",
    ):
        super().__init__()
        self.dtype = dtype
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.num_groups = num_groups

        if isinstance(self.temporal_downsample, int):
            self.temporal_downsample = _get_selected_flags(
                len(self.channel_multipliers) - 1, self.temporal_downsample, False)
            
        self.embedding_dim = latent_embed_dim
        self.conv_downsample = conv_downsample
        self.custom_conv_padding = custom_conv_padding
        # self.norm_type = self.config.vqvae.norm_type
        # self.num_remat_block = self.config.vqvae.get('num_enc_remat_blocks', 0)

        if activation_fn == 'relu':
            self.activation_fn = nn.ReLU
        elif activation_fn == 'swish':
            self.activation_fn = nn.SiLU
        else:
            raise NotImplementedError
        self.activate = self.activation_fn()

        self.conv_fn = functools.partial(
            model_utils.Conv,
            dtype=self.dtype,
            padding='valid' if self.custom_conv_padding is not None else 'same', # SCH: lower letter for pytorch
            custom_padding=self.custom_conv_padding
        )
        
        # self.norm_fn = model_utils.get_norm_layer(
        #     norm_type=self.norm_type, dtype=self.dtype)
        
        self.block_args = dict(
            # norm_fn=self.norm_fn,
            conv_fn=self.conv_fn,
            dtype=self.dtype,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
        )
        

        self.conv1 = self.conv_fn(in_out_channels, self.filters, kernel_size=(3, 3, 3), use_bias=False)

        # ResBlocks and conv downsample
        self.block_res_blocks = []
        self.num_blocks = len(self.channel_multipliers)
        self.conv_blocks = []

        filters = self.filters
        prev_filters = filters # record for in_channels 
        for i in range(self.num_blocks):
            # resblock handling
            filters = self.filters * self.channel_multipliers[i] # SCH: determine the number out_channels
            block_items = []
            for _ in range(self.num_res_blocks):
                block_items.append(ResBlock(prev_filters, filters, **self.block_args))
                prev_filters = filters # update in_channels
            self.block_res_blocks.append(block_items)
            
            if i < self.num_blocks - 1:
                # conv blocks handling
                if self.conv_downsample:
                    t_stride = 2 if self.temporal_downsample[i] else 1
                    self.conv_blocks.append(self.conv_fn(prev_filters, filters, kernel_size=(4, 4, 4), strides=(t_stride, 2, 2))) # SCH: should be same in_channel and out_channel
                    prev_filters = filters # update in_channels

        # NOTE: downsample, dimensions T, H, W
        self.avg_pool_with_t = nn.AvgPool3d((2,2,2))
        self.avg_pool = nn.AvgPool3d((1,2,2))

        # last layer res block
        self.res_blocks = []
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(prev_filters, filters, **self.block_args))
            prev_filters = filters # update in_channels

        # MAGVIT uses Group Normalization
        self.norm1 = nn.GroupNorm(self.num_groups, prev_filters, dtype=dtype) # SCH: separate <prev_filters> channels into 32 groups

        self.conv2 = self.conv_fn(prev_filters, self.embedding_dim, kernel_size=(1, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        for i in range(self.num_blocks):
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)

            if i < self.num_blocks - 1:
                if self.conv_downsample:
                    x = self.conv_blocks[i](x)
                else:
                    if self.temporal_downsample[i]:
                        x = self.avg_pool_with_t(x)
                    else:
                        x = self.avg_pool(x)

        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)

        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x
    


class Decoder(nn.Module):
    """Decoder Blocks."""
    def __init__(self, 
        latent_embed_dim = 256,
        filters = 64,
        in_out_channels = 4,
        num_res_blocks = 2,
        channel_multipliers = (1, 2, 2, 4),
        temporal_downsample = (True, True, False),
        num_groups = 32, # for nn.GroupNorm
        upsample = "nearest+conv", # options: "deconv", "nearest+conv"
        custom_conv_padding = None,
        activation_fn = 'swish',
        dtype="bf16",
    ):

        self.output_dim = in_out_channels
        self.embedding_dim = latent_embed_dim
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.num_groups = num_groups

        if isinstance(self.temporal_downsample, int):
            self.temporal_downsample = _get_selected_flags(
                len(self.channel_multipliers) - 1, self.temporal_downsample, False)
            
        self.upsample = upsample
        self.custom_conv_padding = custom_conv_padding
        # self.norm_type = self.config.vqvae.norm_type
        # self.num_remat_block = self.config.vqvae.get('num_dec_remat_blocks', 0)

        if activation_fn == 'relu':
            self.activation_fn = nn.ReLU
        elif activation_fn == 'swish':
            self.activation_fn = nn.SiLU   
        else:
            raise NotImplementedError  
        self.activate = self.activation_fn()

        self.conv_fn = functools.partial(
            model_utils.Conv,
            dtype=self.dtype,
            padding='VALID' if self.custom_conv_padding is not None else 'SAME',
            custom_padding=self.custom_conv_padding)

        self.conv_t_fn = functools.partial(nn.ConvTranspose3d, dtype=self.dtype)

        # self.norm_fn = model_utils.get_norm_layer(
        #     norm_type=self.norm_type, dtype=self.dtype)

        self.block_args = dict(
            # norm_fn=self.norm_fn,
            conv_fn=self.conv_fn,
            dtype=self.dtype,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
        )
        self.num_blocks = len(self.channel_multipliers)

        filters = self.filters * self.channel_multipliers[-1]

        self.conv1 = self.conv_fn(self.embedding_dim, filters, kernel_size=(3, 3, 3), use_bias=True)

        # last layer res block
        self.res_blocks = []
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(filters, filters, **self.block_args))


        # NOTE: upsample, dimensions T, H, W
        self.upsample_with_t = nn.Upsample(scale_factor=(2,2,2))
        self.upsample = nn.Upsample(scale_factor=(1,2,2))

        # ResBlocks and conv upsample
        prev_filters = filters # SCH: in_channels
        self.block_res_blocks = []
        self.num_blocks = len(self.channel_multipliers)
        self.conv_blocks = []
        # SCH: reverse to keep track of the in_channels, but append also in a reverse direction
        for i in reversed(range(self.num_blocks)): 
            filters = self.filters * self.channel_multipliers[i]
            # resblock handling
            block_items = []
            for _ in range(self.num_res_blocks):
                block_items.append(ResBlock(prev_filters, filters, **self.block_args))
                prev_filters = filters # SCH: update in_channels
            self.block_res_blocks.insert(0, block_items) # SCH: append in front
            
            # conv blocks handling
            if i > 0:
                t_stride = 2 if self.temporal_downsample[i - 1] else 1
                if self.upsample == "deconv":
                    assert self.custom_conv_padding is None, ('Custom padding not implemented for ConvTranspose')
                    # SCH: append in front
                    self.conv_blocks.insert(0, 
                        self.conv_t_fn(prev_filters, filters, kernel_size=(4, 4, 4), strides=(t_stride, 2, 2)))
                    prev_filters = filters # SCH: update in_channels
                elif self.upsample == 'nearest+conv':
                    # SCH: append in front
                    self.conv_blocks.insert(0, self.conv_fn(prev_filters, filters, kernel_size=(3, 3, 3)))
                    prev_filters = filters # SCH: update in_channels
                else:
                    raise NotImplementedError(f'Unknown upsampler: {self.upsample}')
                
        self.norm1 = nn.GroupNorm(self.num_groups, prev_filters, dtyep=dtype)
        self.conv2 = self.conv_fn(prev_filters, self.output_dim, kernel_size=(3, 3, 3))


    def forward(
        self,
        **kwargs,
    ):

        x = self.conv1(x)
        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)
        for i in reversed(range(self.num_blocks)): # reverse here to make decoder symmetric with encoder
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)

            if i > 0:
                if self.upsample == 'deconv':
                    assert self.custom_conv_padding is None, ('Custom padding not implemented for ConvTranspose')
                    x = self.conv_blocks[i-1](x)
            elif self.upsample == 'nearest+conv':
                if self.temporal_downsample[i - 1]:
                    x = self.upsample_with_t(x)
                else:
                    x = self.upsample(x)
                x = self.conv_blocks[i-1](x)
            else:
                raise NotImplementedError(f'Unknown upsampler: {self.upsample}')
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x
    

@MODELS.register_module()
class VAE_3D(nn.Module):
    """The 3D VAE """
    def __init__(
        self, 
        latent_embed_dim = 256,
        filters = 64,
        output_dim: int = 4,
        num_res_blocks = 2,
        channel_multipliers = (1, 2, 2, 4),
        temporal_downsample = (True, True, False),
        num_groups = 32, # for nn.GroupNorm
        conv_downsample = False,
        upsample = "nearest+conv", # options: "deconv", "nearest+conv"
        custom_conv_padding = None,
        activation_fn = 'swish',
        in_out_channels = 4, 
        kl_embed_dim = 64,
        kl_weight = 0.000001,
        dtype="bf16",
        # precision: Any = jax.lax.Precision.DEFAULT
    ):
  

        self.encoder = Encoder(
            filters=filters, 
            num_res_blocks=num_res_blocks, 
            channel_multipliers=channel_multipliers, 
            temporal_downsample=temporal_downsample,
            num_groups = num_groups, # for nn.GroupNorm
            in_out_channels = in_out_channels,
            latent_embed_dim = latent_embed_dim, 
            conv_downsample = conv_downsample, 
            custom_conv_padding = custom_conv_padding,
            activation_fn = activation_fn, 
            dtype=dtype
        )
        self.decoder = Decoder(
            latent_embed_dim = latent_embed_dim,
            filters = filters,
            in_out_channels = in_out_channels, 
            num_res_blocks = num_res_blocks,
            channel_multipliers = channel_multipliers,
            temporal_downsample = temporal_downsample,
            num_groups = num_groups, # for nn.GroupNorm
            upsample = upsample, # options: "deconv", "nearest+conv"
            custom_conv_padding = custom_conv_padding,
            activation_fn = activation_fn,
            dtype=dtype,
        )

        self.loss = model_utils.VEA3DLoss(kl_weight=kl_weight)

        self.quant_conv = nn.Conv3d(2*latent_embed_dim, 2*kl_embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(kl_embed_dim, latent_embed_dim, 1)

    def encode(
        self,
        x,
    ):
        encoded_feature = self.encoder(x)
        moments = self.quant_conv(encoded_feature)
        posterior = model_utils.DiagonalGaussianDistribution(moments)
        return posterior
    
    def decode(
        self,
        z,
    ):  
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    
    def forward(
        self,
        input,
        sample_posterior=True,
    ):  
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior
    

    def get_loss(
        self,
        batch, # [B, C, T, H, W]
        optimizer_idx=None, # NOTE: to add GAN later
    ):
        reconstructions, posterior = self(batch)
        loss = self.loss(batch, reconstructions, posterior)

        return loss 

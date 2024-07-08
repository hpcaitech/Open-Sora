"""
Adapted from SDXL VAE (https://huggingface.co/stabilityai/sdxl-vae/blob/main/config.json)
All default values of kwargs are the same as SDXL
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from einops import rearrange


def video_to_image(func):
    def wrapper(self, x, *args, **kwargs):
        if x.ndim == 5:
            B = x.shape[0]
            x = rearrange(x, 'B C T H W -> (B T) C H W')

            if hasattr(self, 'micro_batch_size') and self.micro_batch_size is None:
                x = func(self, x, *args, **kwargs)
            else:
                bs = self.micro_batch_size
                x_out = []
                for i in range(0, x.shape[0], bs):
                    x_i = func(self, x[i:i + bs], *args, **kwargs)
                    x_out.append(x_i)
                x = torch.cat(x_out, dim=0)

            x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        return x
    return wrapper


class VideoConv2d(nn.Conv2d):
    def __init__(self, *args, micro_batch_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.micro_batch_size = micro_batch_size

    @video_to_image
    def forward(self, x):
        return super().forward(x)


class ResnetBlock2D(nn.Module):
    """
        Use nn.Conv2d
        Default activation is nn.SiLU()
        Make sure input tensor is of shape [B, C, T, H, W] or [B, C, H, W]
        Support micro_batch_size
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        norm_groups: int = 32,
        norm_eps: float = 1e-6,
        micro_batch_size=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.micro_batch_size = micro_batch_size

        conv_cls = nn.Conv2d
        self.norm1 = torch.nn.GroupNorm(num_groups=norm_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.conv1 = conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = torch.nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels, eps=norm_eps, affine=True)
        self.conv2 = conv_cls(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = nn.SiLU()

        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = conv_cls(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    @video_to_image
    def forward(self, x):
        res = self.norm1(x)
        res = self.act(res)
        res = self.conv1(res)

        res = self.norm2(res)
        res = self.act(res)
        res = self.conv2(res)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)

        out = x + res
        return out


class ResnetBlock3D(nn.Module):
    """
        Use nn.Conv3d
        Default activation is nn.SiLU()
        Make sure input tensor is of shape [B, C, T, H, W]
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        norm_groups: int = 32,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        conv_cls = nn.Conv3d
        self.norm1 = torch.nn.GroupNorm(num_groups=norm_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.conv1 = conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = torch.nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels, eps=norm_eps, affine=True)
        self.conv2 = conv_cls(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = nn.SiLU()

        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = conv_cls(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        
    def forward(self, x):
        res = self.norm1(x)
        res = self.act(res)
        res = self.conv1(res)

        res = self.norm2(res)
        res = self.act(res)
        res = self.conv2(res)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)

        out = x + res
        return out


class SpatialDownsample2x(nn.Module):
    """
        Default downsample is Conv2d(stride=2)
        Make sure input tensor is of shape [B, C, T, H, W]
        Support micro_batch_size
    """
    def __init__(
        self,
        channels: int,
        use_conv: bool = True,
        micro_batch_size=None,
    ):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.micro_batch_size = micro_batch_size

        if use_conv:
            self.downsample = nn.Conv2d(
                self.channels, self.channels, kernel_size=3, stride=2, padding=0,
            )
        else:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    @video_to_image
    def forward(self, x):
        # implementation from SDXL
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)

        x = self.downsample(x)
        return x


class SpatialUpsample2x(nn.Module):
    """
        Default upsample is F.interpolate(scale_factor=2) + Conv2d(stride=1)
        Make sure input tensor is of shape [B, C, T, H, W]
        Support micro_batch_size
    """
    def __init__(
        self,
        channels: int,
        use_interpolate=True,
        micro_batch_size=None,
    ):
        super().__init__()
        self.channels = channels
        self.use_interpolate = use_interpolate
        self.micro_batch_size = micro_batch_size

        if use_interpolate:
            self.conv = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1)
        else:
            raise NotImplementedError
            self.upsample = nn.ConvTranspose2d(channels, self.channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        B = x.shape[0]
        x = rearrange(x, 'B C T H W -> (B T) C H W')

        if self.micro_batch_size is None:
            x = self.forward_BCHW(x)
        else:
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_i = self.forward_BCHW(x[i:i + bs])
                x_out.append(x_i)
            x = torch.cat(x_out, dim=0)

        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        return x

    def forward_BCHW(self, x):
        if self.use_interpolate:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if x.shape[0] >= 64:
                x = x.contiguous()

            # interpolate tensor of bfloat16 is fixed in pytorch 2.1. see https://github.com/pytorch/pytorch/issues/86679
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = self.conv(x)
        else:
            x = self.upsample(x)

        return x


class TemporalDownsample2x(nn.Module):
    """
        Default downsample is Conv3d(stride=(2, 1, 1))
        Make sure input tensor is of shape [B, C, T, H, W]
    """
    def __init__(
        self,
        channels: int,
        use_conv: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv

        if use_conv:
            self.downsample = nn.Conv3d(
                self.channels, self.channels, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1),
           )
        else:
            self.downsample = nn.AvgPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1))

    def forward(self, x):
        x = self.downsample(x)
        return x


class TemporalUpsample2x(nn.Module):
    """
        Default upsample is F.interpolate(scale_factor=(2, 1, 1)) + Conv3d(stride=1)
        Make sure input tensor is of shape [B, C, T, H, W]
        Support micro_batch_size
    """
    def __init__(
        self,
        channels,
    ):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        if x.shape[0] >= 64:
            x = x.contiguous()
        x = F.interpolate(x, scale_factor=(2, 1, 1), mode="trilinear")
        x = self.conv(x)
        return x


class UNetMidBlock2D(nn.Module):
    """
        default is ResnetBlock2D + Spatial Attention + ResnetBlock2D
        Make sure input tensor is of shape [B, C, T, H, W] or [B, C, H, W]
    """
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        norm_groups: int = 32,
        norm_eps: float = 1e-6,
        attn_groups: Optional[int] = None,
        add_attention: bool = True,
        attention_head_dim: int = 512,
    ):
        super().__init__()
        self.add_attention = add_attention

        if attn_groups is None:
            attn_groups = norm_groups

        if attention_head_dim is None:
            attention_head_dim = in_channels

        res_blocks = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                norm_eps=norm_eps,
                norm_groups=norm_groups,
            )
        ]
        attn_blocks = []

        for _ in range(num_layers):
            if self.add_attention:
                attn_blocks.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        # rescale_output_factor=output_scale_factor,
                        rescale_output_factor=1.0,
                        eps=norm_eps,
                        norm_num_groups=attn_groups,
                        # spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                        spatial_norm_dim=None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )

            res_blocks.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    norm_eps=norm_eps,
                    norm_groups=norm_groups,
                )
            )

        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.res_blocks = nn.ModuleList(res_blocks)

    def forward(self, x):
        has_T = x.ndim == 5
        if has_T:
            B = x.shape[0]
            x = rearrange(x, 'B C T H W -> (B T) C H W')

        x = self.res_blocks[0](x)
        for attn, res_block in zip(self.attn_blocks, self.res_blocks[1:]):
            if attn is not None:
                x = attn(x)
            x = res_block(x)

        if has_T:
            x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        return x


class Encoder(nn.Module):
    """
        default arch is conv_in + blocks + mid_block + out_block
        Make sure input tensor is of shape [B, C, T, H, W]
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=4,
        norm_groups=32,
        norm_eps=1e-6,
        double_z=True,
        micro_batch_size=None,
    ):
        super().__init__()
        in_channels_encoder = in_channels
        out_channels_encoder = out_channels
        block_out_channels = [128, 256, 512, 512]

        # conv_in
        self.conv_in = VideoConv2d(
            in_channels_encoder,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            micro_batch_size=micro_batch_size,
        )

        # blocks
        blocks = []

        # the first block: ResnetBlock2D
        in_channels = block_out_channels[0]
        out_channels = block_out_channels[0]
        blocks.append(
            nn.Sequential(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    norm_groups=norm_groups,
                    norm_eps=norm_eps,
                    micro_batch_size=micro_batch_size,
                ),
                ResnetBlock2D(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm_groups=norm_groups,
                    norm_eps=norm_eps,
                    micro_batch_size=micro_batch_size,
                ),
                SpatialDownsample2x(
                    channels=out_channels,
                    use_conv=True,
                    micro_batch_size=micro_batch_size, 
                ),
            )
        )

        # the second block: ResnetBlock2D
        in_channels = block_out_channels[0]
        out_channels = block_out_channels[1]
        blocks.append(
            nn.Sequential(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    norm_groups=norm_groups,
                    norm_eps=norm_eps,
                    micro_batch_size=micro_batch_size,
                ),
                ResnetBlock2D(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm_groups=norm_groups,
                    norm_eps=norm_eps,
                    micro_batch_size=micro_batch_size,
                ),
                SpatialDownsample2x(
                    channels=out_channels,
                    use_conv=True,
                    micro_batch_size=micro_batch_size, 
                ),
                TemporalDownsample2x(
                    channels=out_channels,
                    use_conv=True,
                )
            )
        )

        # the third block: ResnetBlock3D
        in_channels = block_out_channels[1]
        out_channels = block_out_channels[2]
        blocks.append(
            nn.Sequential(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    norm_groups=norm_groups,
                    norm_eps=norm_eps,
                ),
                ResnetBlock3D(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm_groups=norm_groups,
                    norm_eps=norm_eps,
                ),
                SpatialDownsample2x(
                    channels=out_channels,
                    use_conv=True,
                ),
                TemporalDownsample2x(
                    channels=out_channels,
                    use_conv=True,
                )
            )
        )

        # the fourth block: ResnetBlock3D
        in_channels = block_out_channels[2]
        out_channels = block_out_channels[3]
        blocks.append(
            nn.Sequential(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    norm_groups=norm_groups,
                    norm_eps=norm_eps,
                ),
                ResnetBlock3D(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm_groups=norm_groups,
                    norm_eps=norm_eps,
                ),
            )
        )

        self.blocks = nn.ModuleList(blocks)


        # mid_block
        in_channels = block_out_channels[-1]
        self.mid_block = UNetMidBlock2D(
            in_channels=in_channels,
            num_layers=1,
            norm_groups=norm_groups,
            norm_eps=norm_eps,
            add_attention=True,
            attention_head_dim=in_channels,
        )

        # out_block
        in_channels = block_out_channels[-1]
        out_channels = 2 * out_channels_encoder if double_z else out_channels_encoder
        self.out_block = nn.Sequential(
            nn.GroupNorm(num_channels=in_channels, num_groups=norm_groups, eps=norm_eps),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        x = self.conv_in(x)

        for block in self.blocks:
            x = block(x)

        x = self.mid_block(x)

        x = self.out_block(x)
        return x


class Decoder(nn.Module):
    """
        default arch is conv_in + mid_block + blocks + out_block
        Make sure input tensor is of shape [B, C, T, H, W]
    """
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        norm_groups=32,
        norm_eps=1e-6,
    ):
        super().__init__()
        in_channels_decoder = in_channels
        out_channels_decoder = out_channels
        block_out_channels = [512, 512, 256, 128]

        # conv_in
        self.conv_in = nn.Conv3d(
            in_channels_decoder,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # mid_block
        in_channels = block_out_channels[0]
        self.mid_block = UNetMidBlock2D(
            in_channels=in_channels,
            num_layers=1,
            norm_groups=norm_groups,
            norm_eps=norm_eps,
            add_attention=True,
            attention_head_dim=in_channels,
        )

        # blocks
        blocks = []
        layer_per_block = 3

        # the first up block: ResnetBlock3D
        in_channels = block_out_channels[0]
        out_channels = block_out_channels[0]
        seq = [
            ResnetBlock3D(
                in_channels=in_channels if idx ==0 else out_channels,
                out_channels=out_channels,
                norm_groups=norm_groups,
                norm_eps=norm_eps,
            )
            for idx in range(layer_per_block)
        ] + [
            SpatialUpsample2x(
                channels=out_channels,
                use_interpolate=True,
            ),
            TemporalUpsample2x(
                channels=out_channels,
            ),
        ]
        blocks.append(nn.Sequential(*seq))

        # the second up block: ResnetBlock3D
        in_channels = block_out_channels[0]
        out_channels = block_out_channels[1]
        seq = [
            ResnetBlock3D(
                in_channels=in_channels if idx ==0 else out_channels,
                out_channels=out_channels,
                norm_groups=norm_groups,
                norm_eps=norm_eps,
            )
            for idx in range(layer_per_block)
        ] + [
            SpatialUpsample2x(
                channels=out_channels,
                use_interpolate=True,
            ),
            TemporalUpsample2x(
                channels=out_channels,
            ),
        ]
        blocks.append(nn.Sequential(*seq))

        # the third up block: ResnetBlock3D
        in_channels = block_out_channels[1]
        out_channels = block_out_channels[2]
        seq = [
            ResnetBlock3D(
                in_channels=in_channels if idx ==0 else out_channels,
                out_channels=out_channels,
                norm_groups=norm_groups,
                norm_eps=norm_eps,
            )
            for idx in range(layer_per_block)
        ] + [
            SpatialUpsample2x(
                channels=out_channels,
                use_interpolate=True,
            ),
        ]
        blocks.append(nn.Sequential(*seq))

        # the fourth up block: ResnetBlock2D
        in_channels = block_out_channels[2]
        out_channels = block_out_channels[3]
        seq = [
            ResnetBlock2D(
                in_channels=in_channels if idx ==0 else out_channels,
                out_channels=out_channels,
                norm_groups=norm_groups,
                norm_eps=norm_eps,
            )
            for idx in range(layer_per_block)
        ]
        blocks.append(nn.Sequential(*seq))

        self.blocks = nn.ModuleList(blocks)

        # out_block
        in_channels = block_out_channels[-1]
        out_channels = out_channels_decoder
        self.out_block = nn.Sequential(
            nn.GroupNorm(num_channels=in_channels, num_groups=norm_groups, eps=norm_eps),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.conv_in(x)
        print(torch.cuda.memory_allocated() /  1024 ** 3)

        x = self.mid_block(x)
        print(torch.cuda.memory_allocated() /  1024 ** 3)

        for block in self.blocks:
            x = block(x)
        print(torch.cuda.memory_allocated() /  1024 ** 3)

        x = self.out_block(x)
        print(torch.cuda.memory_allocated() /  1024 ** 3)
        return x

if __name__ == '__main__':
    from opensora.utils.misc import count_params
    device = 'cuda'
    dtype = torch.bfloat16

    encoder = Encoder(
        in_channels=3,
        out_channels=4,
        double_z=False,
        micro_batch_size=4,
    ).to(torch.bfloat16).to(device, dtype).eval()

    decoder = Decoder(
        in_channels=4,
        out_channels=3,
    ).to(torch.bfloat16).to(device, dtype).eval()
    num_params_enc = count_params(encoder)
    num_params_dec = count_params(decoder)
    print(f'Encoder #params: {num_params_enc}')
    print(f'Decoder #params: {num_params_dec}')

    # inference
    x = torch.rand(1, 3, 51, 720, 1080).to(device, dtype)
    with torch.inference_mode():
        x_enc = encoder(x)
        x_dec = decoder(x_enc)
    print(torch.cuda.memory_allocated() /  1024 ** 3)
    breakpoint()

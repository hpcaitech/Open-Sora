import functools
import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange

from opensora.registry import MODELS
from opensora.utils.ckpt_utils import find_model, load_checkpoint

from .utils import DiagonalGaussianDistribution


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


def pick_video_frame(video, frame_indices):
    """get frame_indices from the video of [B, C, T, H, W] and return images of [B, C, H, W]"""
    batch, device = video.shape[0], video.device
    video = rearrange(video, "b c f ... -> b f c ...")
    batch_indices = torch.arange(batch, device=device)
    batch_indices = rearrange(batch_indices, "b -> b 1")
    images = video[batch_indices, frame_indices]
    images = rearrange(images, "b 1 c ... -> b c ...")
    return images


def exists(v):
    return v is not None


def xavier_uniform_weight_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        # print("initialized module to xavier_uniform:", m)


def Sequential(*modules):
    modules = [*filter(exists, modules)]

    if len(modules) == 0:
        return nn.Identity()

    return nn.Sequential(*modules)


def SameConv2d(dim_in, dim_out, kernel_size):
    kernel_size = cast_tuple(kernel_size, 2)
    padding = [k // 2 for k in kernel_size]
    return nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding)


class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        pad_mode="constant",
        strides=None,  # allow custom stride
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop("dilation", 1)
        stride = strides[0] if strides is not None else kwargs.pop("stride", 1)

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        stride = strides if strides is not None else (stride, 1, 1)
        # padding = kwargs.pop('padding', 0)
        # if padding == "same" and not all([pad == 1 for pad in padding]):
        #     padding = "valid"
        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else "constant"

        x = F.pad(x, self.time_causal_padding, mode=pad_mode)
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,  # SCH: added
        filters,
        conv_fn,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
        num_groups=32,
        device="cpu",
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.activate = activation_fn()
        self.use_conv_shortcut = use_conv_shortcut

        # SCH: MAGVIT uses GroupNorm by default
        self.norm1 = nn.GroupNorm(num_groups, in_channels, device=device, dtype=dtype)
        self.conv1 = conv_fn(in_channels, self.filters, kernel_size=(3, 3, 3), bias=False)
        self.norm2 = nn.GroupNorm(num_groups, self.filters, device=device, dtype=dtype)
        self.conv2 = conv_fn(self.filters, self.filters, kernel_size=(3, 3, 3), bias=False)
        if in_channels != filters:
            if self.use_conv_shortcut:
                self.conv3 = conv_fn(in_channels, self.filters, kernel_size=(3, 3, 3), bias=False)
            else:
                self.conv3 = conv_fn(in_channels, self.filters, kernel_size=(1, 1, 1), bias=False)

    def forward(self, x):
        # device, dtype = x.device, x.dtype
        # input_dim = x.shape[1]
        residual = x
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activate(x)
        x = self.conv2(x)
        if self.in_channels != self.filters:  # SCH: ResBlock X->Y
            residual = self.conv3(residual)
        return x + residual


# SCH: own implementation modified on top of: discriminator with anti-aliased downsampling (blurpool Zhang et al.)
class BlurPool3D(nn.Module):
    def __init__(
        self,
        channels,
        pad_type="reflect",
        filt_size=3,
        stride=2,
        pad_off=0,
        device="cpu",
        dtype=torch.bfloat16,
    ):
        super(BlurPool3D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        if self.filt_size == 1:
            a = np.array(
                [
                    1.0,
                ]
            )
        elif self.filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

        filt_2d = a[:, None] * a[None, :]
        filt_3d = torch.Tensor(a[:, None, None] * filt_2d[None, :, :]).to(device, dtype)

        filt = filt_3d / torch.sum(filt_3d)  # SCH: modified to it 3D
        self.register_buffer("filt", filt[None, None, :, :, :].repeat((self.channels, 1, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, :: self.stride, :: self.stride]
            else:
                return self.pad(inp)[:, :, :: self.stride, :: self.stride]
        else:
            return F.conv3d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if pad_type in ["refl", "reflect"]:
        PadLayer = nn.ReflectionPad3d
    elif pad_type in ["repl", "replicate"]:
        PadLayer = nn.ReplicationPad3d
    elif pad_type == "zero":
        PadLayer = nn.ZeroPad3d
    else:
        print("Pad type [%s] not recognized" % pad_type)
    return PadLayer


class ResBlockDown(nn.Module):
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
        self.conv1 = nn.Conv3d(
            in_channels, in_channels, (3, 3, 3), padding=1, device=device, dtype=dtype
        )  # NOTE: init to xavier_uniform
        self.norm1 = nn.GroupNorm(num_groups, in_channels, device=device, dtype=dtype)

        self.blur = BlurPool3D(in_channels, device=device, dtype=dtype)

        self.conv2 = nn.Conv3d(
            in_channels, self.filters, (1, 1, 1), bias=False, device=device, dtype=dtype
        )  # NOTE: init to xavier_uniform
        self.conv3 = nn.Conv3d(
            in_channels, self.filters, (3, 3, 3), padding=1, device=device, dtype=dtype
        )  # NOTE: init to xavier_uniform
        self.norm2 = nn.GroupNorm(num_groups, self.filters, device=device, dtype=dtype)

        # self.apply(xavier_uniform_weight_init)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation_fn(x)

        residual = self.blur(residual)
        residual = self.conv2(residual)

        x = self.blur(x)
        x = self.conv3(x)
        x = self.norm2(x)
        x = self.activation_fn(x)
        out = (residual + x) / math.sqrt(2)
        return out


# SCH: taken from Open Sora Plan
def n_layer_disc_weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator3D(nn.Module):
    """Defines a 3D PatchGAN discriminator as in Pix2Pix but for 3D inputs."""

    def __init__(self, input_nc=1, ndf=64, n_layers=3, use_actnorm=False):
        """
        Construct a 3D PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input volumes
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            use_actnorm (bool) -- flag to use actnorm instead of batchnorm
        """
        super(NLayerDiscriminator3D, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm3d
        else:
            raise NotImplementedError("Not implemented.")
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=(kw, kw, kw),
                    stride=(1, 2, 2),
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev, ndf * nf_mult, kernel_size=(kw, kw, kw), stride=1, padding=padw, bias=use_bias
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class StyleGANDiscriminatorBlur(nn.Module):
    """StyleGAN Discriminator."""

    """
    SCH: NOTE:
        this discriminator requries the num_frames to be fixed during training;
        in case we pre-train with image then train on video, this disciminator's Linear layer would have to be re-trained!
    """

    def __init__(
        self,
        image_size=(128, 128),
        num_frames=17,
        in_channels=3,
        filters=128,
        channel_multipliers=(2, 4, 4, 4, 4),
        num_groups=32,
        dtype=torch.bfloat16,
        device="cpu",
    ):
        super().__init__()

        self.dtype = dtype
        self.input_size = cast_tuple(image_size, 2)
        self.filters = filters
        self.activation_fn = nn.LeakyReLU(negative_slope=0.2)
        self.channel_multipliers = channel_multipliers

        self.conv1 = nn.Conv3d(
            in_channels, self.filters, (3, 3, 3), padding=1, device=device, dtype=dtype
        )  # NOTE: init to xavier_uniform

        prev_filters = self.filters  # record in_channels
        self.num_blocks = len(self.channel_multipliers)
        self.res_block_list = nn.ModuleList([])
        for i in range(self.num_blocks):
            filters = self.filters * self.channel_multipliers[i]
            self.res_block_list.append(
                ResBlockDown(prev_filters, filters, self.activation_fn, device=device, dtype=dtype).apply(
                    xavier_uniform_weight_init
                )
            )
            prev_filters = filters  # update in_channels

        self.conv2 = nn.Conv3d(
            prev_filters, prev_filters, (3, 3, 3), padding=1, device=device, dtype=dtype
        )  # NOTE: init to xavier_uniform
        # torch.nn.init.xavier_uniform_(self.conv2.weight)

        self.norm1 = nn.GroupNorm(num_groups, prev_filters, dtype=dtype, device=device)

        scale_factor = 2**self.num_blocks
        if num_frames % scale_factor != 0:  # SCH: NOTE: has first frame which would be padded before usage
            time_scaled = num_frames // scale_factor + 1
        else:
            time_scaled = num_frames / scale_factor

        assert (
            self.input_size[0] % scale_factor == 0
        ), f"image width {self.input_size[0]} is not divisible by scale factor {scale_factor}"
        assert (
            self.input_size[1] % scale_factor == 0
        ), f"image height {self.input_size[1]} is not divisible by scale factor {scale_factor}"
        w_scaled, h_scaled = self.input_size[0] / scale_factor, self.input_size[1] / scale_factor
        in_features = int(prev_filters * time_scaled * w_scaled * h_scaled)  # (C*T*W*H)
        self.linear1 = nn.Linear(in_features, prev_filters, device=device, dtype=dtype)  # NOTE: init to xavier_uniform
        self.linear2 = nn.Linear(prev_filters, 1, device=device, dtype=dtype)  # NOTE: init to xavier_uniform

        # self.apply(xavier_uniform_weight_init)

    def forward(self, x):
        x = self.conv1(x)
        # print("discriminator aft conv:", x.size())
        x = self.activation_fn(x)

        for i in range(self.num_blocks):
            x = self.res_block_list[i](x)
            # print("discriminator resblock down:", x.size())

        x = self.conv2(x)
        # print("discriminator aft conv2:", x.size())
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = x.reshape((x.shape[0], -1))  # SCH: [B, (C * T * W * H)] ?

        # print("discriminator reshape:", x.size())
        x = self.linear1(x)
        # print("discriminator aft linear1:", x.size())

        x = self.activation_fn(x)
        x = self.linear2(x)
        # print("discriminator aft linear2:", x.size())
        return x


class Encoder(nn.Module):
    """Encoder Blocks."""

    def __init__(
        self,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
        num_groups=32,  # for nn.GroupNorm
        # in_out_channels = 3, # SCH: added, in_channels at the start
        latent_embed_dim=512,  # num channels for latent vector
        # conv_downsample = False,
        disable_spatial_downsample=False,  # for vae pipeline
        custom_conv_padding=None,
        activation_fn="swish",
        device="cpu",
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.num_groups = num_groups

        self.embedding_dim = latent_embed_dim
        self.disable_spatial_downsample = disable_spatial_downsample
        # self.conv_downsample = conv_downsample
        self.custom_conv_padding = custom_conv_padding

        if activation_fn == "relu":
            self.activation_fn = nn.ReLU
        elif activation_fn == "swish":
            self.activation_fn = nn.SiLU
        else:
            raise NotImplementedError
        self.activate = self.activation_fn()

        self.conv_fn = functools.partial(
            CausalConv3d,
            # padding='valid' if self.custom_conv_padding is not None else 'same', # SCH: lower letter for pytorch
            dtype=dtype,
            device=device,
        )

        self.block_args = dict(
            conv_fn=self.conv_fn,
            dtype=dtype,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
            device=device,
        )

        # NOTE: moved to VAE for separate first frame processing
        # self.conv1 = self.conv_fn(in_out_channels, self.filters, kernel_size=(3, 3, 3), bias=False)

        # ResBlocks and conv downsample
        self.block_res_blocks = nn.ModuleList([])
        self.num_blocks = len(self.channel_multipliers)
        self.conv_blocks = nn.ModuleList([])

        filters = self.filters
        prev_filters = filters  # record for in_channels
        for i in range(self.num_blocks):
            # resblock handling
            filters = self.filters * self.channel_multipliers[i]  # SCH: determine the number out_channels
            block_items = nn.ModuleList([])
            for _ in range(self.num_res_blocks):
                block_items.append(ResBlock(prev_filters, filters, **self.block_args))
                prev_filters = filters  # update in_channels
            self.block_res_blocks.append(block_items)

            if i < self.num_blocks - 1:  # SCH: T-Causal Conv 3x3x3, 128->128, stride t x stride s x stride s
                if self.temporal_downsample[i]: 
                    t_stride = 2 if self.temporal_downsample[i] else 1
                    s_stride = 2 if not self.disable_spatial_downsample else 1
                    self.conv_blocks.append(
                        self.conv_fn(prev_filters, filters, kernel_size=(3, 3, 3), strides=(t_stride, s_stride, s_stride))
                    )  # SCH: should be same in_channel and out_channel
                    prev_filters = filters  # update in_channels
                else: # if no t downsample, don't add since this does nothing for pipeline models
                    self.conv_blocks.append( # Identity
                        nn.Identity(prev_filters)
                    )
                    prev_filters = filters  # update in_channels



        # last layer res block
        self.res_blocks = nn.ModuleList([])
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(prev_filters, filters, **self.block_args))
            prev_filters = filters  # update in_channels

        # MAGVIT uses Group Normalization
        self.norm1 = nn.GroupNorm(
            self.num_groups, prev_filters, dtype=dtype, device=device
        )  # SCH: separate <prev_filters> channels into 32 groups

        self.conv2 = nn.Conv3d(
            prev_filters, self.embedding_dim, kernel_size=(1, 1, 1), dtype=dtype, device=device, padding="same"
        )

    def forward(self, x):
        # dtype, device = x.dtype, x.device

        # NOTE: moved to VAE for separate first frame processing
        # x = self.conv1(x)

        # print("encoder:", x.size())

        try:
            for i in range(self.num_blocks):
                for j in range(self.num_res_blocks):
                    x = self.block_res_blocks[i][j](x)
                    # print("encoder:", x.size())

                if i < self.num_blocks - 1:
                    x = self.conv_blocks[i](x)
                    # print("encoder:", x.size())

            for i in range(self.num_res_blocks):
                x = self.res_blocks[i](x)
                # print("encoder:", x.size())
        except:
            breakpoint()

        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv2(x)
        # print("encoder:", x.size())
        return x


class Decoder(nn.Module):
    """Decoder Blocks."""

    def __init__(
        self,
        latent_embed_dim=512,
        filters=128,
        # in_out_channels = 4,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
        num_groups=32,  # for nn.GroupNorm
        # upsample = "nearest+conv", # options: "deconv", "nearest+conv"
        disable_spatial_upsample=False,  # for vae pipeline
        custom_conv_padding=None,
        activation_fn="swish",
        device="cpu",
        dtype=torch.bfloat16,
    ):
        super().__init__()
        # self.output_dim = in_out_channels
        self.embedding_dim = latent_embed_dim
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.num_groups = num_groups

        # self.upsample = upsample
        self.s_stride = 1 if disable_spatial_upsample else 2  # spatial stride
        self.custom_conv_padding = custom_conv_padding
        # self.norm_type = self.config.vqvae.norm_type
        # self.num_remat_block = self.config.vqvae.get('num_dec_remat_blocks', 0)

        if activation_fn == "relu":
            self.activation_fn = nn.ReLU
        elif activation_fn == "swish":
            self.activation_fn = nn.SiLU
        else:
            raise NotImplementedError
        self.activate = self.activation_fn()

        self.conv_fn = functools.partial(
            CausalConv3d,
            dtype=dtype,
            # padding='valid' if self.custom_conv_padding is not None else 'same', # SCH: lower letter for pytorch
            device=device,
        )

        self.block_args = dict(
            conv_fn=self.conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
            device=device,
            dtype=dtype,
        )
        self.num_blocks = len(self.channel_multipliers)

        filters = self.filters * self.channel_multipliers[-1]

        self.conv1 = self.conv_fn(self.embedding_dim, filters, kernel_size=(3, 3, 3), bias=True)

        # last layer res block
        self.res_blocks = nn.ModuleList([])
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(filters, filters, **self.block_args))

        # TODO: do I need to add adaptive GroupNorm in between each block?

        # # NOTE: upsample, dimensions T, H, W
        # self.upsampler_with_t = nn.Upsample(scale_factor=(2,2,2))
        # self.upsampler = nn.Upsample(scale_factor=(1,2,2))

        # ResBlocks and conv upsample
        prev_filters = filters  # SCH: in_channels
        self.block_res_blocks = nn.ModuleList([])
        self.num_blocks = len(self.channel_multipliers)
        self.conv_blocks = nn.ModuleList([])
        # SCH: reverse to keep track of the in_channels, but append also in a reverse direction
        for i in reversed(range(self.num_blocks)):
            filters = self.filters * self.channel_multipliers[i]
            # resblock handling
            block_items = nn.ModuleList([])
            for _ in range(self.num_res_blocks):
                block_items.append(ResBlock(prev_filters, filters, **self.block_args))
                prev_filters = filters  # SCH: update in_channels
            self.block_res_blocks.insert(0, block_items)  # SCH: append in front

            # conv blocks with upsampling
            if i > 0:
                if self.temporal_downsample[i-1]: 
                    t_stride = 2 if self.temporal_downsample[i - 1] else 1
                    # SCH: T-Causal Conv 3x3x3, f -> (t_stride * 2 * 2) * f, depth to space t_stride x 2 x 2
                    self.conv_blocks.insert(
                        0,
                        self.conv_fn(
                            prev_filters, prev_filters * t_stride * self.s_stride * self.s_stride, kernel_size=(3, 3, 3)
                        ),
                    )
                else:
                    self.conv_blocks.insert(
                        0,
                        nn.Identity(
                            prev_filters
                        ),
                    )


        self.norm1 = nn.GroupNorm(self.num_groups, prev_filters, device=device, dtype=dtype)

        # NOTE: moved to VAE for separate first frame processing
        # self.conv2 = self.conv_fn(prev_filters, self.output_dim, kernel_size=(3, 3, 3))

    def forward(
        self,
        x,
        **kwargs,
    ):
        # dtype, device = x.dtype, x.device

        x = self.conv1(x)
        # print("decoder:", x.size())
        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)
            # print("decoder:", x.size())
        for i in reversed(range(self.num_blocks)):  # reverse here to make decoder symmetric with encoder
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)
                # print("decoder:", x.size())
            if i > 0:
                t_stride = 2 if self.temporal_downsample[i - 1] else 1
                # SCH: T-Causal Conv 3x3x3, f -> (t_stride * 2 * 2) * f, depth to space t_stride x 2 x 2
                x = self.conv_blocks[i - 1](x)
                x = rearrange(
                    x,
                    "B (C ts hs ws) T H W -> B C (T ts) (H hs) (W ws)",
                    ts=t_stride,
                    hs=self.s_stride,
                    ws=self.s_stride,
                )
                # print("decoder:", x.size())

        x = self.norm1(x)
        x = self.activate(x)
        # NOTE: moved to VAE for separate first frame processing
        # x = self.conv2(x)
        return x


@MODELS.register_module()
class VAE_3D_V2(nn.Module):  # , ModelMixin
    """The 3D VAE"""

    def __init__(
        self,
        latent_embed_dim=256,
        filters=128,
        num_res_blocks=2,
        separate_first_frame_encoding=False,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(True, True, False),
        num_groups=32,  # for nn.GroupNorm
        disable_space=False,
        custom_conv_padding=None,
        activation_fn="swish",
        in_out_channels=4,
        kl_embed_dim=64,
        encoder_double_z=True,
        device="cpu",
        dtype="bf16",
    ):
        super().__init__()

        if type(dtype) == str:
            if dtype == "bf16":
                dtype = torch.bfloat16
            elif dtype == "fp16":
                dtype = torch.float16
            else:
                raise NotImplementedError(f"dtype: {dtype}")

        # ==== Model Params ====
        # self.image_size = cast_tuple(image_size, 2)
        self.time_downsample_factor = 2 ** sum(temporal_downsample)
        self.time_padding = self.time_downsample_factor - 1
        self.separate_first_frame_encoding = separate_first_frame_encoding

        image_down = 2 ** len(temporal_downsample)
        t_down = 2 ** len([x for x in temporal_downsample if x == True])
        self.patch_size = (t_down, image_down, image_down)

        # ==== Model Initialization ====

        # encoder & decoder first and last conv layer
        # SCH: NOTE: following MAGVIT, conv in bias=False in encoder first conv
        self.conv_in = CausalConv3d(
            in_out_channels, filters, kernel_size=(3, 3, 3), bias=False, dtype=dtype, device=device
        )
        self.conv_in_first_frame = nn.Identity()
        self.conv_out_first_frame = nn.Identity()
        if separate_first_frame_encoding:
            self.conv_in_first_frame = SameConv2d(in_out_channels, filters, (3, 3))
            self.conv_out_first_frame = SameConv2d(filters, in_out_channels, (3, 3))
        self.conv_out = CausalConv3d(filters, in_out_channels, 3, dtype=dtype, device=device)

        self.encoder = Encoder(
            filters=filters,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            num_groups=num_groups,  # for nn.GroupNorm
            # in_out_channels = in_out_channels,
            latent_embed_dim=latent_embed_dim * 2 if encoder_double_z else latent_embed_dim,
            # conv_downsample = conv_downsample,
            disable_spatial_downsample=disable_space,
            custom_conv_padding=custom_conv_padding,
            activation_fn=activation_fn,
            device=device,
            dtype=dtype,
        )
        self.decoder = Decoder(
            latent_embed_dim=latent_embed_dim,
            filters=filters,
            # in_out_channels = in_out_channels,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            num_groups=num_groups,  # for nn.GroupNorm
            # upsample = upsample, # options: "deconv", "nearest+conv"
            disable_spatial_upsample=disable_space,
            custom_conv_padding=custom_conv_padding,
            activation_fn=activation_fn,
            device=device,
            dtype=dtype,
        )

        if encoder_double_z:
            self.quant_conv = nn.Conv3d(2 * latent_embed_dim, 2 * kl_embed_dim, 1, device=device, dtype=dtype)
        else:
            self.quant_conv = nn.Conv3d(latent_embed_dim, 2 * kl_embed_dim, 1, device=device, dtype=dtype)
        self.post_quant_conv = nn.Conv3d(kl_embed_dim, latent_embed_dim, 1, device=device, dtype=dtype)

    def get_latent_size(self, input_size):
        for i in range(len(input_size)):
            assert input_size[i] % self.patch_size[i] == 0, "Input size must be divisible by patch size"
        input_size = [input_size[i] // self.patch_size[i] for i in range(3)]
        return input_size

    def encode(
        self,
        video,
        video_contains_first_frame=True,
    ):
        self.separate_first_frame_encoding and video_contains_first_frame

        # whether to pad video or not
        # if video_contains_first_frame:
        #     video_len = video.shape[2]
        #     video = pad_at_dim(video, (self.time_padding, 0), value=0.0, dim=2)
        #     video_packed_shape = [torch.Size([self.time_padding]), torch.Size([]), torch.Size([video_len - 1])]

        # print("pre-encoder:", video.size())

        # NOTE: moved encoder conv1 here for separate first frame encoding
        # if encode_first_frame_separately:
        #     pad, first_frame, video = unpack(video, video_packed_shape, "b c * h w")
        #     first_frame = self.conv_in_first_frame(first_frame)
        video = self.conv_in(video)

        # print("pre-encoder:", video.size())

        # if encode_first_frame_separately:
        #     video, _ = pack([first_frame, video], "b c * h w")
        #     video = pad_at_dim(video, (self.time_padding, 0), dim=2)

        encoded_feature = self.encoder(video)

        # print("after encoder:", encoded_feature.size())

        # NOTE: TODO: do we include this before gaussian distri? or go directly to Gaussian distribution
        moments = self.quant_conv(encoded_feature).to(video.dtype)
        posterior = DiagonalGaussianDistribution(moments)

        # print("after encoder moments:", moments.size())

        return posterior

    def decode(
        self,
        z,
        video_contains_first_frame=True,
    ):
        # dtype = z.dtype
        decode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame

        z = self.post_quant_conv(z)
        # print("pre decoder, post quant conv:", z.size())

        dec = self.decoder(z)
        # print("post decoder:", dec.size())

        # SCH: moved decoder last conv layer here for separate first frame decoding
        if decode_first_frame_separately:
            left_pad, dec_ff, dec = (
                dec[:, :, : self.time_padding],
                dec[:, :, self.time_padding],
                dec[:, :, (self.time_padding + 1) :],
            )
            out = self.conv_out(dec)
            outff = self.conv_out_first_frame(dec_ff)
            video, _ = pack([outff, out], "b c * h w")
        else:
            video = self.conv_out(dec)
            # if video were padded, remove padding
            if video_contains_first_frame:
                video = video[:, :, self.time_padding :]

        # print("conv out:", video.size())

        return video

    def get_last_layer(self):
        # CausalConv3d wraps the conv
        return self.conv_out.conv.weight

    def forward(
        self,
        video,
        sample_posterior=True,
        video_contains_first_frame=True,
        # split = "train",
    ):
        batch, channels, frames = video.shape[:3]
        # assert divisible_by(
        #     frames - int(video_contains_first_frame), self.time_downsample_factor
        # ), f"number of frames {frames} minus the first frame ({frames - int(video_contains_first_frame)}) must be divisible by the total downsample factor across time {self.time_downsample_factor}"

        posterior = self.encode(
            video,
            video_contains_first_frame=video_contains_first_frame,
        )

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        recon_video = self.decode(z, video_contains_first_frame=video_contains_first_frame)

        return recon_video, posterior


@MODELS.register_module("VAE_MAGVIT_V2")
def VAE_MAGVIT_V2(from_pretrained=None, **kwargs):
    model = VAE_3D_V2(**kwargs)
    if from_pretrained is not None:
        load_checkpoint(model, from_pretrained, model_name="model")
    return model


@MODELS.register_module("DISCRIMINATOR_3D")
def DISCRIMINATOR_3D(from_pretrained=None, inflate_from_2d=False, use_pretrained=True, **kwargs):
    model = StyleGANDiscriminatorBlur(**kwargs).apply(xavier_uniform_weight_init)
    if from_pretrained is not None:
        if use_pretrained:
            if inflate_from_2d:
                load_checkpoint_with_inflation(model, from_pretrained)
            else:
                load_checkpoint(model, from_pretrained, model_name="discriminator")
                print(f"loaded discriminator")
        else:
            print(f"discriminator use_pretrained={use_pretrained}, initializing new discriminator")

    return model


@MODELS.register_module("N_Layer_DISCRIMINATOR_3D")
def DISCRIMINATOR_3D(from_pretrained=None, inflate_from_2d=False, use_pretrained=True, **kwargs):
    model = NLayerDiscriminator3D(
        input_nc=3,
        n_layers=3,
    ).apply(n_layer_disc_weights_init)
    if from_pretrained is not None:
        if use_pretrained:
            if inflate_from_2d:
                load_checkpoint_with_inflation(model, from_pretrained)
            else:
                load_checkpoint(model, from_pretrained, model_name="discriminator")
                print(f"loaded discriminator")
        else:
            print(f"discriminator use_pretrained={use_pretrained}, initializing new discriminator")

    return model


def load_checkpoint_with_inflation(model, ckpt_path):
    """
    pre-train using image, then inflate to 3D videos
    """
    if ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        state_dict = find_model(ckpt_path)
        with torch.no_grad():
            for key in state_dict:
                if key in model:
                    # central inflation
                    if state_dict[key].size() == model[key][:, :, 0, :, :].size():
                        # temporal dimension
                        val = torch.zeros_like(model[key])
                        centre = int(model[key].size(2) // 2)
                        val[:, :, centre, :, :] = state_dict[key]
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    else:
        load_checkpoint(model, ckpt_path)  # use the default function

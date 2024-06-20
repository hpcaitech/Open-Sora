import functools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opensora.registry import MODELS
from opensora.utils.ckpt_utils import find_model, load_checkpoint


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def xavier_uniform_weight_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        # print("initialized module to xavier_uniform:", m)


# SCH: taken from Open Sora Plan
def n_layer_disc_weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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


@MODELS.register_module()
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False, from_pretrained=None):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        norm_layer = nn.BatchNorm2d

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

        if from_pretrained is not None:
            load_checkpoint(self, from_pretrained)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


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
    """StyleGAN Discriminator.

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


@MODELS.register_module("DISCRIMINATOR_3D")
def DISCRIMINATOR_3D(from_pretrained=None, inflate_from_2d=False, use_pretrained=True, **kwargs):
    model = StyleGANDiscriminatorBlur(**kwargs).apply(xavier_uniform_weight_init)
    if from_pretrained is not None:
        if use_pretrained:
            if inflate_from_2d:
                load_checkpoint_with_inflation(model, from_pretrained)
            else:
                load_checkpoint(model, from_pretrained, model_name="discriminator")
                print("loaded discriminator")
        else:
            print(f"discriminator use_pretrained={use_pretrained}, initializing new discriminator")

    return model


@MODELS.register_module("N_Layer_DISCRIMINATOR_3D")
def DISCRIMINATOR_3D_N_Layer(from_pretrained=None, inflate_from_2d=False, use_pretrained=True, **kwargs):
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
                print("loaded discriminator")
        else:
            print(f"discriminator use_pretrained={use_pretrained}, initializing new discriminator")

    return model

import copy
from pathlib import Path
from math import log2, ceil, sqrt
from functools import wraps, partial

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.autograd import grad as torch_grad

import torchvision
from torchvision.models import VGG16_Weights

from collections import namedtuple

from vector_quantize_pytorch import LFQ, FSQ

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from beartype import beartype
from beartype.typing import Union, Tuple, Optional, List

from magvit2_pytorch.attend import Attend
from magvit2_pytorch.version import __version__

from gateloop_transformer import SimpleGateLoopLayer

from taylor_series_linear_attention import TaylorSeriesLinearAttn

from kornia.filters import filter3d

import pickle

# helper

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def safe_get_index(it, ind, default = None):
    if ind < len(it):
        return it[ind]
    return default

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def identity(t, *args, **kwargs):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def append_dims(t, ndims: int):
    return t.reshape(*t.shape, *((1,) * ndims))

def is_odd(n):
    return not divisible_by(n, 2)

def maybe_del_attr_(o, attr):
    if hasattr(o, attr):
        delattr(o, attr)

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# tensor helpers

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pick_video_frame(video, frame_indices):
    batch, device = video.shape[0], video.device
    video = rearrange(video, 'b c f ... -> b f c ...')
    batch_indices = torch.arange(batch, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')
    images = video[batch_indices, frame_indices]
    images = rearrange(images, 'b 1 c ... -> b c ...')
    return images

# gan related

def gradient_penalty(images, output):
    batch_size = images.shape[0]

    gradients = torch_grad(
        outputs = output,
        inputs = images,
        grad_outputs = torch.ones(output.size(), device = images.device),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return ((gradients.norm(2, dim = 1) - 1) ** 2).mean()

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

@autocast(enabled = False)
@beartype
def grad_layer_wrt_loss(
    loss: Tensor,
    layer: nn.Parameter
):
    return torch_grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True
    )[0].detach()

# helper decorators

def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, 'vgg')
        if has_vgg:
            vgg = self.vgg
            delattr(self, 'vgg')

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self.vgg = vgg

        return out
    return inner

# helper classes

def Sequential(*modules):
    modules = [*filter(exists, modules)]

    if len(modules) == 0:
        return nn.Identity()

    return nn.Sequential(*modules)

class Residual(Module):
    @beartype
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# for a bunch of tensor operations to change tensor to (batch, time, feature dimension) and back

class ToTimeSequence(Module):
    @beartype
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = rearrange(x, 'b c f ... -> b ... f c')
        x, ps = pack_one(x, '* n c')

        o = self.fn(x, **kwargs)

        o = unpack_one(o, ps, '* n c')
        return rearrange(o, 'b ... f c -> b c f ...')


class SqueezeExcite(Module):
    # global context network - attention-esque squeeze-excite variant (https://arxiv.org/abs/2012.13375)

    def __init__(
        self,
        dim,
        *,
        dim_out = None,
        dim_hidden_min = 16,
        init_bias = -10
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.to_k = nn.Conv2d(dim, 1, 1)
        dim_hidden = max(dim_hidden_min, dim_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(dim, dim_hidden, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim_hidden, dim_out, 1),
            nn.Sigmoid()
        )

        nn.init.zeros_(self.net[-2].weight)
        nn.init.constant_(self.net[-2].bias, init_bias)

    def forward(self, x):
        orig_input, batch = x, x.shape[0]
        is_video = x.ndim == 5

        if is_video:
            x = rearrange(x, 'b c f h w -> (b f) c h w')

        context = self.to_k(x)

        context = rearrange(context, 'b c h w -> b c (h w)').softmax(dim = -1)
        spatial_flattened_input = rearrange(x, 'b c h w -> b c (h w)')

        out = einsum('b i n, b c n -> b c i', context, spatial_flattened_input)
        out = rearrange(out, '... -> ... 1')
        gates = self.net(out)

        if is_video:
            gates = rearrange(gates, '(b f) c h w -> b c f h w', b = batch)

        return gates * orig_input

# token shifting

class TokenShift(Module):
    @beartype
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        x, x_shift = x.chunk(2, dim = 1)
        x_shift = pad_at_dim(x_shift, (1, -1), dim = 2) # shift time dimension
        x = torch.cat((x, x_shift), dim = 1)
        return self.fn(x, **kwargs)

# rmsnorm

class RMSNorm(Module):
    def __init__(
        self,
        dim,
        channel_first = False,
        images = False,
        bias = False
    ):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(x, dim = (1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias

class AdaptiveRMSNorm(Module):
    def __init__(
        self,
        dim,
        *,
        dim_cond,
        channel_first = False,
        images = False,
        bias = False
    ):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.dim_cond = dim_cond
        self.channel_first = channel_first
        self.scale = dim ** 0.5

        self.to_gamma = nn.Linear(dim_cond, dim)
        self.to_bias = nn.Linear(dim_cond, dim) if bias else None

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)

        if bias:
            nn.init.zeros_(self.to_bias.weight)
            nn.init.zeros_(self.to_bias.bias)

    @beartype
    def forward(self, x: Tensor, *, cond: Tensor):
        batch = x.shape[0]
        assert cond.shape == (batch, self.dim_cond)

        gamma = self.to_gamma(cond)

        bias = 0.
        if exists(self.to_bias):
            bias = self.to_bias(cond)

        if self.channel_first:
            gamma = append_dims(gamma, x.ndim - 2)

            if exists(self.to_bias):
                bias = append_dims(bias, x.ndim - 2)

        return F.normalize(x, dim = (1 if self.channel_first else -1)) * self.scale * gamma + bias

# attention

class Attention(Module):
    @beartype
    def __init__(
        self,
        *,
        dim,
        dim_cond: Optional[int] = None,
        causal = False,
        dim_head = 32,
        heads = 8,
        flash = False,
        dropout = 0.,
        num_memory_kv = 4
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.need_cond = exists(dim_cond)

        if self.need_cond:
            self.norm = AdaptiveRMSNorm(dim, dim_cond = dim_cond)
        else:
            self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        )

        assert num_memory_kv > 0
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_memory_kv, dim_head))

        self.attend = Attend(
            causal = causal,
            dropout = dropout,
            flash = flash
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False)
        )

    @beartype
    def forward(
        self,
        x,
        mask: Optional[Tensor ] = None,
        cond: Optional[Tensor] = None
    ):
        maybe_cond_kwargs = dict(cond = cond) if self.need_cond else dict()

        x = self.norm(x, **maybe_cond_kwargs)

        q, k, v = self.to_qkv(x)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = q.shape[0]), self.mem_kv)
        k = torch.cat((mk, k), dim = -2)
        v = torch.cat((mv, v), dim = -2)

        out = self.attend(q, k, v, mask = mask)
        return self.to_out(out)

class LinearAttention(Module):
    """
    using the specific linear attention proposed in https://arxiv.org/abs/2106.09681
    """

    @beartype
    def __init__(
        self,
        *,
        dim,
        dim_cond: Optional[int] = None,
        dim_head = 8,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.need_cond = exists(dim_cond)

        if self.need_cond:
            self.norm = AdaptiveRMSNorm(dim, dim_cond = dim_cond)
        else:
            self.norm = RMSNorm(dim)

        self.attn = TaylorSeriesLinearAttn(
            dim = dim,
            dim_head = dim_head,
            heads = heads
        )

    def forward(
        self,
        x,
        cond: Optional[Tensor] = None
    ):
        maybe_cond_kwargs = dict(cond = cond) if self.need_cond else dict()

        x = self.norm(x, **maybe_cond_kwargs)

        return self.attn(x)

class LinearSpaceAttention(LinearAttention):
    """
    SCH: format h & w into a linear dim, do linear attention, then format back
    """
    def forward(self, x, *args, **kwargs):
        x = rearrange(x, 'b c ... h w -> b ... h w c')
        x, batch_ps = pack_one(x, '* h w c')
        x, seq_ps = pack_one(x, 'b * c')

        x = super().forward(x, *args, **kwargs)

        x = unpack_one(x, seq_ps, 'b * c')
        x = unpack_one(x, batch_ps, '* h w c')
        return rearrange(x, 'b ... h w c -> b c ... h w')

class SpaceAttention(Attention):
    def forward(self, x, *args, **kwargs):
        x = rearrange(x, 'b c t h w -> b t h w c')
        x, batch_ps = pack_one(x, '* h w c')
        x, seq_ps = pack_one(x, 'b * c')

        x = super().forward(x, *args, **kwargs)

        x = unpack_one(x, seq_ps, 'b * c')
        x = unpack_one(x, batch_ps, '* h w c')
        return rearrange(x, 'b t h w c -> b c t h w')

class TimeAttention(Attention):
    def forward(self, x, *args, **kwargs):
        x = rearrange(x, 'b c t h w -> b h w t c')
        x, batch_ps = pack_one(x, '* t c')

        x = super().forward(x, *args, **kwargs)

        x = unpack_one(x, batch_ps, '* t c')
        return rearrange(x, 'b h w t c -> b c t h w')

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = 1)
        return F.gelu(gate) * x

class FeedForward(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        dim_cond: Optional[int] = None,
        mult = 4,
        images = False
    ):
        super().__init__()
        conv_klass = nn.Conv2d if images else nn.Conv3d

        rmsnorm_klass = RMSNorm if not exists(dim_cond) else partial(AdaptiveRMSNorm, dim_cond = dim_cond)

        maybe_adaptive_norm_klass = partial(rmsnorm_klass, channel_first = True, images = images)

        dim_inner = int(dim * mult * 2 / 3)

        self.norm = maybe_adaptive_norm_klass(dim)

        self.net = Sequential(
            conv_klass(dim, dim_inner * 2, 1),
            GEGLU(),
            conv_klass(dim_inner, dim, 1)
        )

    @beartype
    def forward(
        self,
        x: Tensor,
        *,
        cond: Optional[Tensor] = None
    ):
        maybe_cond_kwargs = dict(cond = cond) if exists(cond) else dict()

        x = self.norm(x, **maybe_cond_kwargs)
        return self.net(x)

# discriminator with anti-aliased downsampling (blurpool Zhang et al.)

class Blur(Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(
        self,
        x,
        space_only = False,
        time_only = False
    ):
        assert not (space_only and time_only)

        f = self.f

        if space_only:
            f = einsum('i, j -> i j', f, f)
            f = rearrange(f, '... -> 1 1 ...')
        elif time_only:
            f = rearrange(f, 'f -> 1 f 1 1')
        else:
            f = einsum('i, j, k -> i j k', f, f, f)
            f = rearrange(f, '... -> 1 ...')

        is_images = x.ndim == 4

        if is_images:
            x = rearrange(x, 'b c h w -> b c 1 h w')

        out = filter3d(x, f, normalized = True)

        if is_images:
            out = rearrange(out, 'b c 1 h w -> b c h w')

        return out

class DiscriminatorBlock(Module):
    def __init__(
        self,
        input_channels,
        filters,
        downsample = True,
        antialiased_downsample = True
    ):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding = 1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding = 1),
            leaky_relu()
        )

        self.maybe_blur = Blur() if antialiased_downsample else None

        self.downsample = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
            nn.Conv2d(filters * 4, filters, 1)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)

        x = self.net(x)

        if exists(self.downsample):
            if exists(self.maybe_blur):
                x = self.maybe_blur(x, space_only = True)

            x = self.downsample(x)

        x = (x + res) * (2 ** -0.5)
        return x

class Discriminator(Module):
    @beartype
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels = 3,
        max_dim = 512,
        attn_heads = 8,
        attn_dim_head = 32,
        linear_attn_dim_head = 8,
        linear_attn_heads = 16,
        ff_mult = 4,
        antialiased_downsample = False
    ):
        super().__init__()
        image_size = pair(image_size)
        min_image_resolution = min(image_size)

        num_layers = int(log2(min_image_resolution) - 2)

        blocks = []

        layer_dims = [channels] + [(dim * 4) * (2 ** i) for i in range(num_layers + 1)] # SCH: num channels across each layer
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        blocks = []
        attn_blocks = []

        image_resolution = min_image_resolution

        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            block = DiscriminatorBlock(
                in_chan,
                out_chan,
                downsample = is_not_last,
                antialiased_downsample = antialiased_downsample
            )

            attn_block = Sequential(
                Residual(LinearSpaceAttention(
                    dim = out_chan,
                    heads = linear_attn_heads,
                    dim_head = linear_attn_dim_head
                )),
                Residual(FeedForward(
                    dim = out_chan,
                    mult = ff_mult,
                    images = True
                ))
            )

            blocks.append(ModuleList([
                block,
                attn_block
            ]))

            image_resolution //= 2

        self.blocks = ModuleList(blocks)

        dim_last = layer_dims[-1]

        downsample_factor = 2 ** num_layers
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))

        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        self.to_logits = Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding = 1),
            leaky_relu(),
            Rearrange('b ... -> b (...)'),
            nn.Linear(latent_dim, 1),
            Rearrange('b 1 -> b')
        )

    def forward(self, x):

        for block, attn_block in self.blocks:
            x = block(x)
            x = attn_block(x)

        return self.to_logits(x)

# modulatable conv from Karras et al. Stylegan2
# for conditioning on latents

class Conv3DMod(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        spatial_kernel,
        time_kernel,
        causal = True,
        dim_out = None,
        demod = True,
        eps = 1e-8,
        pad_mode = 'zeros'
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.eps = eps

        assert is_odd(spatial_kernel) and is_odd(time_kernel)

        self.spatial_kernel = spatial_kernel
        self.time_kernel = time_kernel

        time_padding = (time_kernel - 1, 0) if causal else ((time_kernel // 2,) * 2)

        self.pad_mode = pad_mode
        self.padding = (*((spatial_kernel // 2,) * 4), *time_padding)
        self.weights = nn.Parameter(torch.randn((dim_out, dim, time_kernel, spatial_kernel, spatial_kernel)))

        self.demod = demod

        nn.init.kaiming_normal_(self.weights, a = 0, mode = 'fan_in', nonlinearity = 'selu')

    @beartype
    def forward(
        self,
        fmap,
        cond: Tensor
    ):
        """
        notation

        b - batch
        n - convs
        o - output
        i - input
        k - kernel
        """

        b = fmap.shape[0]

        # prepare weights for modulation

        weights = self.weights

        # do the modulation, demodulation, as done in stylegan2

        cond = rearrange(cond, 'b i -> b 1 i 1 1 1')

        weights = weights * (cond + 1)

        if self.demod:
            inv_norm = reduce(weights ** 2, 'b o i k0 k1 k2 -> b o 1 1 1 1', 'sum').clamp(min = self.eps).rsqrt()
            weights = weights * inv_norm

        fmap = rearrange(fmap, 'b c t h w -> 1 (b c) t h w')

        weights = rearrange(weights, 'b o ... -> (b o) ...')

        fmap = F.pad(fmap, self.padding, mode = self.pad_mode)
        fmap = F.conv3d(fmap, weights, groups = b)

        return rearrange(fmap, '1 (b o) ... -> b o ...', b = b)

# strided conv downsamples

class SpatialDownsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        kernel_size = 3,
        antialias = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.maybe_blur = Blur() if antialias else identity
        self.conv = nn.Conv2d(dim, dim_out, kernel_size, stride = 2, padding = kernel_size // 2)

    def forward(self, x):
        x = self.maybe_blur(x, space_only = True)

        x = rearrange(x, 'b c t h w -> b t c h w')
        x, ps = pack_one(x, '* c h w')

        out = self.conv(x)

        out = unpack_one(out, ps, '* c h w')
        out = rearrange(out, 'b t c h w -> b c t h w')
        return out

class TimeDownsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        kernel_size = 3,
        antialias = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.maybe_blur = Blur() if antialias else identity
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(dim, dim_out, kernel_size, stride = 2)

    def forward(self, x):
        x = self.maybe_blur(x, time_only = True)

        x = rearrange(x, 'b c t h w -> b h w c t')
        x, ps = pack_one(x, '* c t')

        x = F.pad(x, self.time_causal_padding)
        out = self.conv(x)

        out = unpack_one(out, ps, '* c t')
        out = rearrange(out, 'b h w c t -> b c t h w')
        return out

# depth to space upsamples

class SpatialUpsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange('b (c p1 p2) h w -> b c (h p1) (w p2)', p1 = 2, p2 = 2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b t c h w')
        x, ps = pack_one(x, '* c h w')

        out = self.net(x)

        out = unpack_one(out, ps, '* c h w')
        out = rearrange(out, 'b t c h w -> b c t h w')
        return out

class TimeUpsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv1d(dim, dim_out * 2, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange('b (c p) t -> b c (t p)', p = 2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, t = conv.weight.shape
        conv_weight = torch.empty(o // 2, i, t)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 2) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b h w c t')
        x, ps = pack_one(x, '* c t')

        out = self.net(x)

        out = unpack_one(out, ps, '* c t')
        out = rearrange(out, 'b h w c t -> b c t h w')
        return out

# autoencoder - only best variant here offered, with causal conv 3d

def SameConv2d(dim_in, dim_out, kernel_size):
    kernel_size = cast_tuple(kernel_size, 2)
    padding = [k // 2 for k in kernel_size]
    return nn.Conv2d(dim_in, dim_out, kernel_size = kernel_size, padding = padding)

class CausalConv3d(Module):
    @beartype
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        pad_mode = 'constant',
        **kwargs
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop('dilation', 1)
        stride = kwargs.pop('stride', 1)

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride = stride, dilation = dilation, **kwargs)

    def forward(self, x):
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else 'constant'

        x = F.pad(x, self.time_causal_padding, mode = pad_mode)
        return self.conv(x)

@beartype
def ResidualUnit(
    dim,
    kernel_size: Union[int, Tuple[int, int, int]],
    pad_mode: str = 'constant'
):
    net = Sequential(
        CausalConv3d(dim, dim, kernel_size, pad_mode = pad_mode),
        nn.ELU(),
        nn.Conv3d(dim, dim, 1),
        nn.ELU(),
        SqueezeExcite(dim)
    )

    return Residual(net)

@beartype
class ResidualUnitMod(Module):
    def __init__(
        self,
        dim,
        kernel_size: Union[int, Tuple[int, int, int]],
        *,
        dim_cond,
        pad_mode: str = 'constant',
        demod = True
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        assert height_kernel_size == width_kernel_size

        self.to_cond = nn.Linear(dim_cond, dim)

        self.conv = Conv3DMod(
            dim = dim,
            spatial_kernel = height_kernel_size,
            time_kernel = time_kernel_size,
            causal = True,
            demod = demod,
            pad_mode = pad_mode
        )

        self.conv_out = nn.Conv3d(dim, dim, 1)

    @beartype
    def forward(
        self,
        x,
        cond: Tensor,
    ):
        res = x
        cond = self.to_cond(cond)

        x = self.conv(x, cond = cond)
        x = F.elu(x)
        x = self.conv_out(x)
        x = F.elu(x)
        return x + res

class CausalConvTranspose3d(Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        *,
        time_stride,
        **kwargs
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        self.upsample_factor = time_stride

        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        stride = (time_stride, 1, 1)
        padding = (0, height_pad, width_pad)

        self.conv = nn.ConvTranspose3d(chan_in, chan_out, kernel_size, stride, padding = padding, **kwargs)

    def forward(self, x):
        assert x.ndim == 5
        t = x.shape[2]

        out = self.conv(x)

        out = out[..., :(t * self.upsample_factor), :, :]
        return out

# video tokenizer class

LossBreakdown = namedtuple('LossBreakdown', [
    'recon_loss',
    'lfq_aux_loss',
    'quantizer_loss_breakdown',
    'perceptual_loss',
    'adversarial_gen_loss',
    'adaptive_adversarial_weight',
    'multiscale_gen_losses',
    'multiscale_gen_adaptive_weights'
])

DiscrLossBreakdown = namedtuple('DiscrLossBreakdown', [
    'discr_loss',
    'multiscale_discr_losses',
    'gradient_penalty'
])

class VideoTokenizer(Module):
    @beartype
    def __init__(
        self,
        *,
        image_size,
        layers: Tuple[Union[str, Tuple[str, int]], ...] = (
            'residual',
            'residual',
            'residual'
        ),
        residual_conv_kernel_size = 3,
        # num_codebooks = 1,
        # codebook_size: Optional[int] = None,
        channels = 3,
        init_dim = 64,
        max_dim = float('inf'),
        dim_cond = None,
        dim_cond_expansion_factor = 4.,
        input_conv_kernel_size: Tuple[int, int, int] = (7, 7, 7),
        output_conv_kernel_size: Tuple[int, int, int] = (3, 3, 3),
        pad_mode: str = 'constant',
        lfq_entropy_loss_weight = 0.1,
        lfq_commitment_loss_weight = 1., # SCH: codebook?
        lfq_diversity_gamma = 2.5,
        quantizer_aux_loss_weight = 1.,
        lfq_activation = nn.Identity(),
        use_fsq = False,
        fsq_levels: Optional[List[int]] = None,
        attn_dim_head = 32,
        attn_heads = 8,
        attn_dropout = 0.,
        linear_attn_dim_head = 8,
        linear_attn_heads = 16,
        vgg: Optional[Module] = None,
        vgg_weights: VGG16_Weights = VGG16_Weights.DEFAULT,
        perceptual_loss_weight = 1e-1,
        discr_kwargs: Optional[dict] = None,
        multiscale_discrs: Tuple[Module, ...] = tuple(),
        use_gan = True,
        adversarial_loss_weight = 1.,
        grad_penalty_loss_weight = 10.,
        multiscale_adversarial_loss_weight = 1.,
        flash_attn = True,
        separate_first_frame_encoding = False
    ):
        super().__init__()

        # for autosaving the config

        _locals = locals()
        _locals.pop('self', None)
        _locals.pop('__class__', None)
        self._configs = pickle.dumps(_locals)

        # image size

        self.channels = channels
        self.image_size = image_size

        # initial encoder

        self.conv_in = CausalConv3d(channels, init_dim, input_conv_kernel_size, pad_mode = pad_mode)

        # whether to encode the first frame separately or not

        self.conv_in_first_frame = nn.Identity()
        self.conv_out_first_frame = nn.Identity()

        if separate_first_frame_encoding:
            self.conv_in_first_frame = SameConv2d(channels, init_dim, input_conv_kernel_size[-2:])
            self.conv_out_first_frame = SameConv2d(init_dim, channels, output_conv_kernel_size[-2:])

        self.separate_first_frame_encoding = separate_first_frame_encoding

        # encoder and decoder layers

        self.encoder_layers = ModuleList([])
        self.decoder_layers = ModuleList([])

        self.conv_out = CausalConv3d(init_dim, channels, output_conv_kernel_size, pad_mode = pad_mode)

        dim = init_dim
        dim_out = dim

        layer_fmap_size = image_size # SCH: feaure map size
        time_downsample_factor = 1
        has_cond_across_layers = [] # SCH: record if the corr. layers has condition

        for layer_def in layers:
            layer_type, *layer_params = cast_tuple(layer_def)

            has_cond = False

            if layer_type == 'residual': # SCH: resblock
                encoder_layer = ResidualUnit(dim, residual_conv_kernel_size)
                decoder_layer = ResidualUnit(dim, residual_conv_kernel_size)

            elif layer_type == 'consecutive_residual':
                num_consecutive, = layer_params
                encoder_layer = Sequential(*[ResidualUnit(dim, residual_conv_kernel_size) for _ in range(num_consecutive)])
                decoder_layer = Sequential(*[ResidualUnit(dim, residual_conv_kernel_size) for _ in range(num_consecutive)])

            elif layer_type == 'cond_residual':
                assert exists(dim_cond), 'dim_cond must be passed into VideoTokenizer, if tokenizer is to be conditioned'

                has_cond = True

                encoder_layer = ResidualUnitMod(dim, residual_conv_kernel_size, dim_cond = int(dim_cond * dim_cond_expansion_factor))
                decoder_layer = ResidualUnitMod(dim, residual_conv_kernel_size, dim_cond = int(dim_cond * dim_cond_expansion_factor))
                dim_out = dim

            elif layer_type == 'compress_space':
                dim_out = safe_get_index(layer_params, 0)
                dim_out = default(dim_out, dim * 2) # SCH: if dim_out exists, else use dim * 2
                dim_out = min(dim_out, max_dim)

                encoder_layer = SpatialDownsample2x(dim, dim_out) # SCH: 2d conv in space dimensions
                decoder_layer = SpatialUpsample2x(dim_out, dim) # SCH: 2d conv in space dimensions, use more channel to expand space dim

                assert layer_fmap_size > 1
                layer_fmap_size //= 2

            elif layer_type == 'compress_time':
                dim_out = safe_get_index(layer_params, 0)
                dim_out = default(dim_out, dim * 2)
                dim_out = min(dim_out, max_dim)

                encoder_layer = TimeDownsample2x(dim, dim_out) # SCH: 1d conv in time dim to reduce
                decoder_layer = TimeUpsample2x(dim_out, dim) # SCH: 1d conv in time dim, use more channels to expand time dim

                time_downsample_factor *= 2

            elif layer_type == 'attend_space':
                attn_kwargs = dict(
                    dim = dim,
                    dim_head = attn_dim_head,
                    heads = attn_heads,
                    dropout = attn_dropout,
                    flash = flash_attn
                )

                encoder_layer = Sequential(
                    Residual(SpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim))
                )

                decoder_layer = Sequential(
                    Residual(SpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim))
                )

            elif layer_type == 'linear_attend_space':
                linear_attn_kwargs = dict(
                    dim = dim,
                    dim_head = linear_attn_dim_head,
                    heads = linear_attn_heads
                )

                encoder_layer = Sequential(
                    Residual(LinearSpaceAttention(**linear_attn_kwargs)),
                    Residual(FeedForward(dim))
                )

                decoder_layer = Sequential(
                    Residual(LinearSpaceAttention(**linear_attn_kwargs)),
                    Residual(FeedForward(dim))
                )

            elif layer_type == 'gateloop_time':
                gateloop_kwargs = dict(
                    use_heinsen = False
                )

                encoder_layer = ToTimeSequence(Residual(SimpleGateLoopLayer(dim = dim)))
                decoder_layer = ToTimeSequence(Residual(SimpleGateLoopLayer(dim = dim)))

            elif layer_type == 'attend_time':
                attn_kwargs = dict(
                    dim = dim,
                    dim_head = attn_dim_head,
                    heads = attn_heads,
                    dropout = attn_dropout,
                    causal = True,
                    flash = flash_attn
                )

                encoder_layer = Sequential(
                    Residual(TokenShift(TimeAttention(**attn_kwargs))),
                    Residual(TokenShift(FeedForward(dim, dim_cond = dim_cond)))
                )

                decoder_layer = Sequential(
                    Residual(TokenShift(TimeAttention(**attn_kwargs))),
                    Residual(TokenShift(FeedForward(dim, dim_cond = dim_cond)))
                )

            elif layer_type == 'cond_attend_space':
                has_cond = True

                attn_kwargs = dict(
                    dim = dim,
                    dim_cond = dim_cond,
                    dim_head = attn_dim_head,
                    heads = attn_heads,
                    dropout = attn_dropout,
                    flash = flash_attn
                )

                encoder_layer = Sequential(
                    Residual(SpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim))
                )

                decoder_layer = Sequential(
                    Residual(SpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim))
                )

            elif layer_type == 'cond_linear_attend_space':
                has_cond = True

                attn_kwargs = dict(
                    dim = dim,
                    dim_cond = dim_cond,
                    dim_head = attn_dim_head,
                    heads = attn_heads,
                    dropout = attn_dropout,
                    flash = flash_attn
                )

                encoder_layer = Sequential(
                    Residual(LinearSpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim, dim_cond = dim_cond))
                )

                decoder_layer = Sequential(
                    Residual(LinearSpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim, dim_cond = dim_cond))
                )

            elif layer_type == 'cond_attend_time':
                has_cond = True

                attn_kwargs = dict(
                    dim = dim,
                    dim_cond = dim_cond,
                    dim_head = attn_dim_head,
                    heads = attn_heads,
                    dropout = attn_dropout,
                    causal = True,
                    flash = flash_attn
                )

                encoder_layer = Sequential(
                    Residual(TokenShift(TimeAttention(**attn_kwargs))),
                    Residual(TokenShift(FeedForward(dim, dim_cond = dim_cond)))
                )

                decoder_layer = Sequential(
                    Residual(TokenShift(TimeAttention(**attn_kwargs))),
                    Residual(TokenShift(FeedForward(dim, dim_cond = dim_cond)))
                )

            else:
                raise ValueError(f'unknown layer type {layer_type}')

            self.encoder_layers.append(encoder_layer)
            self.decoder_layers.insert(0, decoder_layer)

            dim = dim_out
            has_cond_across_layers.append(has_cond)

        # add a final norm just before quantization layer

        self.encoder_layers.append(Sequential(
            Rearrange('b c ... -> b ... c'),
            nn.LayerNorm(dim),
            Rearrange('b ... c -> b c ...'),
        ))

        self.time_downsample_factor = time_downsample_factor
        self.time_padding = time_downsample_factor - 1

        self.fmap_size = layer_fmap_size

        # use a MLP stem for conditioning, if needed

        self.has_cond_across_layers = has_cond_across_layers
        self.has_cond = any(has_cond_across_layers)

        self.encoder_cond_in = nn.Identity()
        self.decoder_cond_in = nn.Identity()

        if has_cond:
            self.dim_cond = dim_cond

            self.encoder_cond_in = Sequential(
                nn.Linear(dim_cond, int(dim_cond * dim_cond_expansion_factor)),
                nn.SiLU()
            )

            self.decoder_cond_in = Sequential(
                nn.Linear(dim_cond, int(dim_cond * dim_cond_expansion_factor)),
                nn.SiLU()
            )

        ## SCH: remove quantizer
        # # quantizer related

        # self.use_fsq = use_fsq

        # if not use_fsq:
        #     assert exists(codebook_size) and not exists(fsq_levels), 'if use_fsq is set to False, `codebook_size` must be set (and not `fsq_levels`)'

        #     # lookup free quantizer(s) - multiple codebooks is possible
        #     # each codebook will get its own entropy regularization

        #     self.quantizers = LFQ(
        #         dim = dim,
        #         codebook_size = codebook_size,
        #         num_codebooks = num_codebooks,
        #         entropy_loss_weight = lfq_entropy_loss_weight,
        #         commitment_loss_weight = lfq_commitment_loss_weight,
        #         diversity_gamma = lfq_diversity_gamma
        #     )

        # else:
        #     assert not exists(codebook_size) and exists(fsq_levels), 'if use_fsq is set to True, `fsq_levels` must be set (and not `codebook_size`). the effective codebook size is the cumulative product of all the FSQ levels'

        #     self.quantizers = FSQ(
        #         fsq_levels,
        #         dim = dim,
        #         num_codebooks = num_codebooks
        #     )

        # self.quantizer_aux_loss_weight = quantizer_aux_loss_weight

        # dummy loss

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # perceptual loss related

        use_vgg = channels in {1, 3, 4} and perceptual_loss_weight > 0.

        self.vgg = None
        self.perceptual_loss_weight = perceptual_loss_weight

        if use_vgg:
            if not exists(vgg):
                vgg = torchvision.models.vgg16(
                    weights = vgg_weights
                )

                vgg.classifier = Sequential(*vgg.classifier[:-2])

            self.vgg = vgg

        self.use_vgg = use_vgg

        # main flag for whether to use GAN at all

        self.use_gan = use_gan

        # discriminator

        discr_kwargs = default(discr_kwargs, dict(
            dim = dim,
            image_size = image_size,
            channels = channels,
            max_dim = 512
        ))

        self.discr = Discriminator(**discr_kwargs)

        self.adversarial_loss_weight = adversarial_loss_weight
        self.grad_penalty_loss_weight = grad_penalty_loss_weight

        self.has_gan = use_gan and adversarial_loss_weight > 0.

        # multi-scale discriminators

        self.has_multiscale_gan = use_gan and multiscale_adversarial_loss_weight > 0.

        self.multiscale_discrs = ModuleList([*multiscale_discrs])

        self.multiscale_adversarial_loss_weight = multiscale_adversarial_loss_weight

        self.has_multiscale_discrs = (
            use_gan and \
            multiscale_adversarial_loss_weight > 0. and \
            len(multiscale_discrs) > 0
        )

    @property
    def device(self):
        return self.zero.device

    @classmethod
    def init_and_load_from(cls, path, strict = True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        assert 'config' in pkg, 'model configs were not found in this saved checkpoint'

        config = pickle.loads(pkg['config'])
        tokenizer = cls(**config)
        tokenizer.load(path, strict = strict)
        return tokenizer

    def parameters(self):
        return [
            *self.conv_in.parameters(),
            *self.conv_in_first_frame.parameters(),
            *self.conv_out_first_frame.parameters(),
            *self.conv_out.parameters(),
            *self.encoder_layers.parameters(),
            *self.decoder_layers.parameters(),
            *self.encoder_cond_in.parameters(),
            *self.decoder_cond_in.parameters(),
            *self.quantizers.parameters()
        ]

    def discr_parameters(self):
        return self.discr.parameters()

    def copy_for_eval(self):
        device = self.device
        vae_copy = copy.deepcopy(self.cpu())

        maybe_del_attr_(vae_copy, 'discr')
        maybe_del_attr_(vae_copy, 'vgg')
        maybe_del_attr_(vae_copy, 'multiscale_discrs')

        vae_copy.eval()
        return vae_copy.to(device)

    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists(), f'{str(path)} already exists'

        pkg = dict(
            model_state_dict = self.state_dict(),
            version = __version__,
            config = self._configs
        )

        torch.save(pkg, str(path))

    def load(self, path, strict = True):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path))
        state_dict = pkg.get('model_state_dict')
        version = pkg.get('version')

        assert exists(state_dict)

        if exists(version):
            print(f'loading checkpointed tokenizer from version {version}')

        self.load_state_dict(state_dict, strict = strict)

    @beartype
    def encode(
        self,
        video: Tensor,
        quantize = False,
        cond: Optional[Tensor] = None,
        video_contains_first_frame = True
    ):
        """
        SCH: conv (may sep 1st frame), then pass through self.encoder_layers, then quantize if needed, finish
        """
        encode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame

        # whether to pad video or not

        if video_contains_first_frame:
            video_len = video.shape[2]

            video = pad_at_dim(video, (self.time_padding, 0), value = 0., dim = 2)
            video_packed_shape = [torch.Size([self.time_padding]), torch.Size([]), torch.Size([video_len - 1])]

        # conditioning, if needed

        assert (not self.has_cond) or exists(cond), '`cond` must be passed into tokenizer forward method since conditionable layers were specified'

        if exists(cond):
            assert cond.shape == (video.shape[0], self.dim_cond)

            cond = self.encoder_cond_in(cond)
            cond_kwargs = dict(cond = cond)

        # initial conv
        # taking into account whether to encode first frame separately

        if encode_first_frame_separately:
            pad, first_frame, video = unpack(video, video_packed_shape, 'b c * h w')
            first_frame = self.conv_in_first_frame(first_frame)

        video = self.conv_in(video)

        if encode_first_frame_separately:
            video, _ = pack([first_frame, video], 'b c * h w')
            video = pad_at_dim(video, (self.time_padding, 0), dim = 2)

        # encoder layers

        for fn, has_cond in zip(self.encoder_layers, self.has_cond_across_layers):

            layer_kwargs = dict()

            if has_cond:
                layer_kwargs = cond_kwargs

            video = fn(video, **layer_kwargs)

        maybe_quantize = identity if not quantize else self.quantizers

        return maybe_quantize(video)

    # @beartype
    # def decode_from_code_indices(
    #     self,
    #     codes: Tensor,
    #     cond: Optional[Tensor] = None,
    #     video_contains_first_frame = True
    # ):
    #     assert codes.dtype in (torch.long, torch.int32)

    #     if codes.ndim == 2:
    #         video_code_len = codes.shape[-1]
    #         assert divisible_by(video_code_len, self.fmap_size ** 2), f'flattened video ids must have a length ({video_code_len}) that is divisible by the fmap size ({self.fmap_size}) squared ({self.fmap_size ** 2})'

    #         codes = rearrange(codes, 'b (f h w) -> b f h w', h = self.fmap_size, w = self.fmap_size)

    #     quantized = self.quantizers.indices_to_codes(codes)

    #     return self.decode(quantized, cond = cond, video_contains_first_frame = video_contains_first_frame)

    @beartype
    def decode(
        self,
        quantized: Tensor,
        cond: Optional[Tensor] = None,
        video_contains_first_frame = True
    ):
        decode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame

        batch = quantized.shape[0]

        # conditioning, if needed

        assert (not self.has_cond) or exists(cond), '`cond` must be passed into tokenizer forward method since conditionable layers were specified'

        if exists(cond): # SCH: quantized latents used as control signal following StyleGAN?
            assert cond.shape == (batch, self.dim_cond)

            cond = self.decoder_cond_in(cond) # SCH: linear + activation
            cond_kwargs = dict(cond = cond)

        # decoder layers

        x = quantized

        for fn, has_cond in zip(self.decoder_layers, reversed(self.has_cond_across_layers)):

            layer_kwargs = dict()

            if has_cond:
                layer_kwargs = cond_kwargs

            x = fn(x, **layer_kwargs)

        # to pixels

        if decode_first_frame_separately:
            left_pad, xff, x = x[:, :, :self.time_padding], x[:, :, self.time_padding], x[:, :, (self.time_padding + 1):]

            out = self.conv_out(x)
            outff = self.conv_out_first_frame(xff)

            video, _ = pack([outff, out], 'b c * h w')

        else:
            video = self.conv_out(x)

            # if video were padded, remove padding

            if video_contains_first_frame:
                video = video[:, :, self.time_padding:]

        return video

    @torch.no_grad()
    def tokenize(self, video):
        self.eval()
        return self.forward(video, return_codes = True)

    @beartype
    def forward(
        self,
        video_or_images: Tensor,
        cond: Optional[Tensor] = None,
        return_loss = False,
        return_codes = False,
        return_recon = False,
        return_discr_loss = False,
        return_recon_loss_only = False,
        apply_gradient_penalty = True,
        video_contains_first_frame = True,
        adversarial_loss_weight = None,
        multiscale_adversarial_loss_weight = None
    ):
        adversarial_loss_weight = default(adversarial_loss_weight, self.adversarial_loss_weight)
        multiscale_adversarial_loss_weight = default(multiscale_adversarial_loss_weight, self.multiscale_adversarial_loss_weight)

        assert (return_loss + return_codes + return_discr_loss) <= 1
        assert video_or_images.ndim in {4, 5}

        assert video_or_images.shape[-2:] == (self.image_size, self.image_size)

        # accept images for image pretraining (curriculum learning from images to video)

        is_image = video_or_images.ndim == 4

        if is_image:
            video = rearrange(video_or_images, 'b c ... -> b c 1 ...')
            video_contains_first_frame = True
        else:
            video = video_or_images

        batch, channels, frames = video.shape[:3]

        assert divisible_by(frames - int(video_contains_first_frame), self.time_downsample_factor), f'number of frames {frames} minus the first frame ({frames - int(video_contains_first_frame)}) must be divisible by the total downsample factor across time {self.time_downsample_factor}'

        # encoder

        x = self.encode(video, cond = cond, video_contains_first_frame = video_contains_first_frame)


        ## SCH: remove the codebook
        # # lookup free quantization

        # if self.use_fsq:
        #     quantized, codes = self.quantizers(x)

        #     aux_losses = self.zero
        #     quantizer_loss_breakdown = None
        # else:
        #     (quantized, codes, aux_losses), quantizer_loss_breakdown = self.quantizers(x, return_loss_breakdown = True)

        # if return_codes and not return_recon:
        #     return codes

        # decoder
        recon_video = self.decode(x, cond = cond, video_contains_first_frame = video_contains_first_frame)

        # if return_codes:
        #     return codes, recon_video

        # reconstruction loss

        if not (return_loss or return_discr_loss or return_recon_loss_only):
            return recon_video

        recon_loss = F.mse_loss(video, recon_video)

        # for validation, only return recon loss

        if return_recon_loss_only:
            return recon_loss, recon_video

        # TODO:
        # gan discriminator loss

        if return_discr_loss:
            assert self.has_gan
            assert exists(self.discr)

            # pick a random frame for image discriminator

            frame_indices = torch.randn((batch, frames)).topk(1, dim = -1).indices

            real = pick_video_frame(video, frame_indices)

            if apply_gradient_penalty:
                real = real.requires_grad_()

            fake = pick_video_frame(recon_video, frame_indices)

            real_logits = self.discr(real)
            fake_logits = self.discr(fake.detach())

            discr_loss = hinge_discr_loss(fake_logits, real_logits)

            # multiscale discriminators

            multiscale_discr_losses = []

            if self.has_multiscale_discrs:
                for discr in self.multiscale_discrs:
                    multiscale_real_logits = discr(video)
                    multiscale_fake_logits = discr(recon_video.detach())

                    multiscale_discr_loss = hinge_discr_loss(multiscale_fake_logits, multiscale_real_logits)

                    multiscale_discr_losses.append(multiscale_discr_loss)
            else:
                multiscale_discr_losses.append(self.zero)

            # gradient penalty

            if apply_gradient_penalty:
                gradient_penalty_loss = gradient_penalty(real, real_logits)
            else:
                gradient_penalty_loss = self.zero

            # total loss

            total_loss = discr_loss + \
                gradient_penalty_loss * self.grad_penalty_loss_weight + \
                sum(multiscale_discr_losses) * self.multiscale_adversarial_loss_weight

            discr_loss_breakdown = DiscrLossBreakdown(
                discr_loss,
                multiscale_discr_losses,
                gradient_penalty_loss
            )

            return total_loss, discr_loss_breakdown

        # perceptual loss

        if self.use_vgg:

            frame_indices = torch.randn((batch, frames)).topk(1, dim = -1).indices

            input_vgg_input = pick_video_frame(video, frame_indices)
            recon_vgg_input = pick_video_frame(recon_video, frame_indices)

            if channels == 1:
                input_vgg_input = repeat(input_vgg_input, 'b 1 h w -> b c h w', c = 3)
                recon_vgg_input = repeat(recon_vgg_input, 'b 1 h w -> b c h w', c = 3)

            elif channels == 4:
                input_vgg_input = input_vgg_input[:, :3]
                recon_vgg_input = recon_vgg_input[:, :3]

            input_vgg_feats = self.vgg(input_vgg_input)
            recon_vgg_feats = self.vgg(recon_vgg_input)

            perceptual_loss = F.mse_loss(input_vgg_feats, recon_vgg_feats)
        else:
            perceptual_loss = self.zero

        # get gradient with respect to perceptual loss for last decoder layer
        # needed for adaptive weighting

        last_dec_layer = self.conv_out.conv.weight

        norm_grad_wrt_perceptual_loss = None

        if self.training and self.use_vgg and (self.has_gan or self.has_multiscale_discrs):
            norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

        # per-frame image discriminator

        recon_video_frames = None

        if self.has_gan:
            frame_indices = torch.randn((batch, frames)).topk(1, dim = -1).indices
            recon_video_frames = pick_video_frame(recon_video, frame_indices)

            fake_logits = self.discr(recon_video_frames)
            gen_loss = hinge_gen_loss(fake_logits)

            adaptive_weight = 1.

            if exists(norm_grad_wrt_perceptual_loss):
                norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
                adaptive_weight = norm_grad_wrt_perceptual_loss / norm_grad_wrt_gen_loss.clamp(min = 1e-3)
                adaptive_weight.clamp_(max = 1e3)

                if torch.isnan(adaptive_weight).any():
                    adaptive_weight = 1.
        else:
            gen_loss = self.zero
            adaptive_weight = 0.

        # multiscale discriminator losses

        multiscale_gen_losses = []
        multiscale_gen_adaptive_weights = []

        if self.has_multiscale_gan and self.has_multiscale_discrs:
            if not exists(recon_video_frames):
                recon_video_frames = pick_video_frame(recon_video, frame_indices)

            for discr in self.multiscale_discrs:
                fake_logits = recon_video_frames
                multiscale_gen_loss = hinge_gen_loss(fake_logits)

                multiscale_gen_losses.append(multiscale_gen_loss)

                multiscale_adaptive_weight = 1.

                if exists(norm_grad_wrt_perceptual_loss):
                    norm_grad_wrt_gen_loss = grad_layer_wrt_loss(multiscale_gen_loss, last_dec_layer).norm(p = 2)
                    multiscale_adaptive_weight = norm_grad_wrt_perceptual_loss / norm_grad_wrt_gen_loss.clamp(min = 1e-5)
                    multiscale_adaptive_weight.clamp_(max = 1e3)

                multiscale_gen_adaptive_weights.append(multiscale_adaptive_weight)

        # calculate total loss

        total_loss = recon_loss \
            + aux_losses * self.quantizer_aux_loss_weight \
            + perceptual_loss * self.perceptual_loss_weight \
            + gen_loss * adaptive_weight * adversarial_loss_weight

        if self.has_multiscale_discrs:

            weighted_multiscale_gen_losses = sum(loss * weight for loss, weight in zip(multiscale_gen_losses, multiscale_gen_adaptive_weights))

            total_loss = total_loss + weighted_multiscale_gen_losses * multiscale_adversarial_loss_weight

        # loss breakdown

        loss_breakdown = LossBreakdown(
            recon_loss,
            aux_losses,
            quantizer_loss_breakdown,
            perceptual_loss,
            gen_loss,
            adaptive_weight,
            multiscale_gen_losses,
            multiscale_gen_adaptive_weights
        )

        return total_loss, loss_breakdown

# main class
class MagViT2(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

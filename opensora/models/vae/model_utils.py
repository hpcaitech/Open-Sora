import functools
import math
from typing import Any, Optional, Sequence, Type

import torch.nn as nn
import numpy as np
import torch


## NOTE: not used since we only have 'GN'
# def get_norm_layer(norm_type, dtype):
#   if norm_type == 'LN':
#     # supply a few args with partial function and pass the rest of the args when this norm_fn is called
#     norm_fn = functools.partial(nn.LayerNorm, dtype=dtype)
#   elif norm_type == 'GN': # 
#     norm_fn = functools.partial(nn.GroupNorm, dtype=dtype)
#   elif norm_type is None:
#     norm_fn = lambda: (lambda x: x)
#   else:
#     raise NotImplementedError(f'norm_type: {norm_type}')
#   return norm_fn

## NOTE: not used since need to put nn.AvgPool3d in init
# def downsample(x, include_t_dim: bool = True, factor: int = 2):
#     """Downsample via average pooling."""
#     t_factor = factor if include_t_dim else 1
#     shape = (t_factor, factor, factor)
#     average_pool = nn.AvgPool3d(shape) # x shape needs to be [N,C,D,H,W]
#     x = average_pool(x)
#     return x

## NOTE: not used since need to put nn.Upsample in init
# def upsample(x: jnp.ndarray, include_t_dim: bool = True, factor: int = 2):
#   """Upsample via nearest interpolation."""
#   n, t, h, w, c = x.shape
#   upsample = nn.Upsample(scale_factor=(factor if include_t_dim else 1, factor, factor))
#   x = upsample(x)
#   return x

# class Conv(nn.Conv3d):
#     """Convolution with custom padding.

#     Attributes:
#         custom_padding: padding mode accepted by jnp.pad. When using this, must set
#         padding=VALID to disable padding in nn.Conv.
#     """

#     def __init__(
#         self, 
#         in_channels,
#         out_channels,
#         kernel_size,
#         dtype = "bf16",
#         padding = "same",
#         use_bias=False,
#         custom_padding:Optional[str] = None,
#     ):
#         super(Conv, self).__init__(in_channels, out_channels, kernel_size, dtype=dtype, padding=padding)
#         self.custom_padding = custom_padding

#     def forward(self, x):
#         if self.custom_padding is not None:
#             assert self.padding == 'valid', 'Must use valid padding for raw Conv.'
#             assert self.dilation == 1, 'Kernel dilation not supported.'
#             pads = [((k - 1) // 2, k // 2) for k in self.kernel_size]
#             pads = [(0, 0)] + pads + [(0, 0)]
#             if self.custom_padding.startswith('reflect_') \
#                 or self.custom_padding.startswith('symmetric_'):
#                 custom_padding, reflect_type = self.custom_padding.split('_')
#                 pad_kwargs = {'reflect_type': reflect_type}
#             else:
#                 custom_padding = self.custom_padding
#                 pad_kwargs = {}
 
#             x = np.pad(x, pads, mode=custom_padding, **pad_kwargs)
#             return super(Conv, self).__call__(x)
        

class DiagonalGaussianDistribution(object):
    def __init__(
        self, 
        parameters, 
        deterministic=False,
    ):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1) # SCH: channels dim
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3, 4]) # TODO: check dimensions
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3, 4]) # TODO: check dimensions

    def nll(self, sample, dims=[1,2,3,4]): # TODO: check dimensions
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
    
class VEA3DLoss(nn.Module):
    def __init__(
        self,
        # disc_start, 
        logvar_init=0.0, 
        kl_weight=1.0, 
        pixelloss_weight=1.0,
        disc_num_layers=3, 
        disc_in_channels=3, 
        disc_factor=1.0, 
        disc_weight=1.0,
        perceptual_weight=1.0, 
        use_actnorm=False, 
        disc_conditional=False,
        disc_loss="hinge"
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        # self.perceptual_loss = LPIPS().eval() # TODO
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        # self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
        #                                          n_layers=disc_num_layers,
        #                                          use_actnorm=use_actnorm
        #                                          ).apply(weights_init)
        # self.discriminator_iter_start = disc_start
        # self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        # self.disc_factor = disc_factor
        # self.discriminator_weight = disc_weight
        # self.disc_conditional = disc_conditional

    # TODO: for discriminator
    # def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
    #     if last_layer is not None:
    #         nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    #         g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    #     else:
    #         nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
    #         g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

    #     d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    #     d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    #     d_weight = d_weight * self.discriminator_weight
    #     return d_weight
    
    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        # optimizer_idx,
        # global_step, 
        last_layer=None, 
        cond=None, 
        split="train",
        weights=None,
    ):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        # if self.perceptual_weight > 0: # TODO
        #     p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
        #     rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss = weighted_nll_loss + self.kl_weight * kl_loss # TODO: add discriminator loss later 

        # log = {"{}/total_loss".format(split): loss.clone().detach().mean(), 
        #        "{}/logvar".format(split): self.logvar.detach(),
        #        "{}/kl_loss".format(split): kl_loss.detach().mean(), 
        #        "{}/nll_loss".format(split): nll_loss.detach().mean(),
        #        "{}/rec_loss".format(split): rec_loss.detach().mean(),
        #     #    "{}/d_weight".format(split): d_weight.detach(),
        #     #    "{}/disc_factor".format(split): torch.tensor(disc_factor),
        #     #    "{}/g_loss".format(split): g_loss.detach().mean(),
        #     }
        
        return loss


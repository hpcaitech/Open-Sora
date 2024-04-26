import numpy as np
import torch
import torch.nn as nn

# from taming.modules.losses.lpips import LPIPS # need to pip install https://github.com/CompVis/taming-transformers
# from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from einops import rearrange

"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""


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


class DiagonalGaussianDistribution(object):
    def __init__(
        self,
        parameters,
        deterministic=False,
    ):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)  # SCH: channels dim
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device, dtype=self.mean.dtype)

    def sample(self):
        # torch.randn: standard normal distribution
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device, dtype=self.mean.dtype)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:  # SCH: assumes other is a standard normal distribution
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3, 4])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3, 4],
                )

    def nll(self, sample, dims=[1, 2, 3, 4]):  # TODO: what does this do?
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):  # SCH: used for vae inference?
        return self.mean


class VEA3DLoss(nn.Module):
    def __init__(
        self,
        # disc_start,
        logvar_init=0.0,
        kl_weight=1.0,
        pixelloss_weight=1.0,
        perceptual_weight=0.1,
        disc_loss="hinge",
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        # optimizer_idx,
        # global_step,
        weights=None,
    ):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:  # NOTE: need in_channels == 3 in order to use!
            assert inputs.size(1) == 3, f"using vgg16 that requires 3 input channels but got {inputs.size(1)}"
            # SCH: transform to [(B,T), C, H, W] shape for percetual loss over each frame
            B = inputs.shape[0]
            inputs = rearrange(inputs, "B C T H W -> (B T) C H W")
            reconstructions = rearrange(reconstructions, "B C T H W -> (B T) C H W")
            # permutated_input = torch.permute(inputs, (0, 2, 1, 3, 4)) # [B, C, T, H, W] --> [B, T, C, H, W]
            # permutated_rec = torch.permute(reconstructions, (0, 2, 1, 3, 4))
            # data_shape = permutated_input.size()
            # p_loss = self.perceptual_loss(
            #     permutated_input.reshape(-1, data_shape[-3], data_shape[-2],data_shape[-1]).contiguous(),
            #     permutated_rec.reshape(-1, data_shape[-3], data_shape[-2],data_shape[-1]).contiguous()
            # )
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            # SCH: shape back p_loss
            # permuted_p_loss = torch.permute(p_loss.reshape(data_shape[0], data_shape[1], 1, 1, 1), (0,2,1,3,4))
            # rec_loss = rec_loss + self.perceptual_weight * permuted_p_loss
            p_loss = rearrange(p_loss, "(B T) C H W -> B C T H W", B=B)
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss = weighted_nll_loss + self.kl_weight * kl_loss  # TODO: add discriminator loss later

        return loss


class VEA3DLossWithDiscriminator(nn.Module):
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
        disc_loss="hinge",
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
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
        if self.perceptual_weight > 0:  # NOTE: need in_channels == 3 in order to use!
            assert inputs.size(1) == 3, f"using vgg16 that requires 3 input channels but got {inputs.size(1)} "
            # SCH: transform to [(B,T), C, H, W] shape for percetual loss over each frame
            permutated_input = torch.permute(inputs, (0, 2, 1, 3, 4))  # [B, C, T, H, W] --> [B, T, C, H, W]
            permutated_rec = torch.permute(reconstructions, (0, 2, 1, 3, 4))
            data_shape = permutated_input.size()
            p_loss = self.perceptual_loss(
                permutated_input.reshape(-1, data_shape[-3], data_shape[-2], data_shape[-1]).contiguous(),
                permutated_rec.reshape(-1, data_shape[-3], data_shape[-2], data_shape[-1]).contiguous(),
            )
            # SCH: shape back p_loss
            permuted_p_loss = torch.permute(p_loss.reshape(data_shape[0], data_shape[1], 1, 1, 1), (0, 2, 1, 3, 4))
            rec_loss = rec_loss + self.perceptual_weight * permuted_p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss = weighted_nll_loss + self.kl_weight * kl_loss  # TODO: add discriminator loss later

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

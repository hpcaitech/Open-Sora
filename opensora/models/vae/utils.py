import numpy as np
import torch

# from taming.modules.losses.lpips import LPIPS # need to pip install https://github.com/CompVis/taming-transformers
# from taming.modules.discriminator.model import NLayerDiscriminator, weights_init

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
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
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

    def mode(self):
        return self.mean

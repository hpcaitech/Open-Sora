import torch
import numpy as np
from typing import Union
from einops import rearrange
from typing import List
from ..iddpm.gaussian_diffusion import _extract_into_tensor, mean_flat

# some code are inspired by https://github.com/magic-research/piecewise-rectified-flow/blob/main/scripts/train_perflow.py
# and https://github.com/magic-research/piecewise-rectified-flow/blob/main/src/scheduler_perflow.py


class RFlowScheduler:
    def __init__(
        self,
        num_timesteps = 1000,
        num_sampling_steps = 10,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        

    def training_losses(self, model, x_start, t, model_kwargs=None, noise = None, mask = None, weights = None):
        '''
        Compute training losses for a single timestep. 
        Arguments format copied from opensora/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        '''

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(x_start, noise, t)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)


        terms = {}
        model_output = model(x_t, t, **model_kwargs)
        velocity_pred = model_output.chunk(2, dim = 1)[0]
        if weights is None:
            loss = mean_flat((velocity_pred - (x_start - noise)).pow(2), mask = mask)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(weight * (velocity_pred - (x_start - noise)).pow(2), mask = mask)
        terms['loss'] = loss

        return terms


    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps # [0, 999/1000]
        timepoints = 1 - timepoints # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])

        return timepoints * original_samples + (1 - timepoints) * noise
    


        
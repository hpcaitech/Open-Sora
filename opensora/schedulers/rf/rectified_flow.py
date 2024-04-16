import torch
import numpy as np
from typing import Union

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
        assert mask is None, "mask not support for rectified flow yet"
        assert weights is None, "weights not support for rectified flow yet"

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(x_start, noise, t)

        terms = {}
        model_output = model(x_t, t, **model_kwargs)
        velocity_pred = model_output.chunk(2, dim = 1)[0]
        loss = (velocity_pred - (x_start - noise)).pow(2).mean()
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
    
    # def step(
    #     self,
    #     model_output: torch.FloatTensor,
    #     timestep: Union[int, torch.IntTensor],
    #     sample: torch.FloatTensor,
    # ) -> torch.FloatTensor:
    #     '''
    #     take an Euler step sampling
    #     '''

    #     dt = 1 / self.num_sampling_steps

    #     return sample + dt * model_output


        
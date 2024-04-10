import torch
import numpy as np

# some code are inspired by https://github.com/magic-research/piecewise-rectified-flow/blob/main/scripts/train_perflow.py


class RFlowScheduler:
    def __init__(
        self,
        num_timesteps = 1000,
    ):
        self.num_timesteps = num_timesteps
        

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

        x_t = self.add_noise(x_start, noise, t)

        terms = {}
        velocity_pred = model(x_t, t, **model_kwargs)
        loss = (velocity_pred - (x_start - noise)).pow(2).mean()
        terms['loss'] = loss

        return terms

    def add_noise(self, x0, x1, t):
        '''
        x0: sample of dataset
        x1: sample of gaussian distribution
        '''
        # rescale t from [0,num_timesteps] to [0,1]
        t = t / self.num_timesteps
        return t * x1 + (1 - t) * x0
    
    def step():
        pass
        
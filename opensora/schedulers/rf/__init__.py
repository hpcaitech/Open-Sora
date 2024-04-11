# should have property num_timesteps, 
# method sample()  training_losses()
import torch
from .rectified_flow import RFlowScheduler
from functools import partial

from opensora.registry import SCHEDULERS

@SCHEDULERS.register_module("rflow")
class RFLOW:
    def __init__(self, num_sampling_steps = 10, num_timesteps = 1000, cfg_scale = 4.0):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale

        self.scheduler = RFlowScheduler(num_timesteps = num_timesteps, num_sampling_steps = num_sampling_steps)

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale = None,
        # progress = True,
    ):
        assert mask is None, "mask is not supported in rectified flow yet"
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        z = torch.cat([z, z], 0)
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        timesteps = [(1. - i/self.num_sampling_steps) * 1000. for i in range(self.num_sampling_steps)]

        # convert float timesteps to most close int timesteps
        timesteps = [int(round(t)) for t in timesteps]

        for i, t in enumerate(timesteps):
            pred = model(z, torch.tensor(t * z.shape[0], device = device), **model_args)
            pred_cond, pred_uncond = pred.chunk(2, dim = 1)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            dt = (timesteps[i] - timesteps[i+1])/self.num_timesteps if i < len(timesteps) - 1 else 1/self.num_timesteps
            z = z + v_pred * dt

        return z

        
        

    
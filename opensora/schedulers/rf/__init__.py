from functools import partial

import torch

from opensora.registry import SCHEDULERS

from .rectified_flow import RFlowScheduler, timestep_transform


@SCHEDULERS.register_module("rflow")
class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            **kwargs,
        )

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
    ):
        assert mask is None, "mask is not supported in rectified flow inference yet"
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        timesteps = [(1.0 - i / self.num_sampling_steps) * 1000.0 for i in range(self.num_sampling_steps)]

        # convert float timesteps to most close int timesteps
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]

        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, scale=self.cfg_scale) for t in timesteps]

        for i, t in enumerate(timesteps):
            z_in = torch.cat([z, z], 0)
            pred = model(z_in, torch.tensor([t] * z_in.shape[0], device=device), **model_args).chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            dt = (
                (timesteps[i] - timesteps[i + 1]) / self.num_timesteps
                if i < len(timesteps) - 1
                else 1 / self.num_timesteps
            )
            z = z + v_pred * dt

        return z

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None):
        return self.scheduler.training_losses(model, x_start, model_kwargs, noise, mask, weights)

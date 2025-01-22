import random

import torch

from ..iddpm.gaussian_diffusion import _extract_into_tensor, mean_flat
from .time_sampler import TimeSampler

# some code are inspired by https://github.com/magic-research/piecewise-rectified-flow/blob/main/scripts/train_perflow.py
# and https://github.com/magic-research/piecewise-rectified-flow/blob/main/src/scheduler_perflow.py


class RFlowScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        # time sampler
        sample_method="uniform",
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        transform_scale=1.0,
        scale_temporal=True,
        uniform_over_threshold=None,
        drop_condition=None,
        x_cond_weight=1,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.time_sampler = TimeSampler(
            sample_method=sample_method,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            transform_scale=transform_scale,
            scale_temporal=scale_temporal,
            uniform_over_threshold=uniform_over_threshold,
        )
        self.drop_condition = drop_condition
        self.x_cond_weight = x_cond_weight  # the weight to use for i2v and v2v condition

    def training_losses(
        self,
        model,
        x_start,
        model_kwargs=None,
        noise=None,
        mask=None,
        weights=None,
        t=None,
        x_gt=None,
        mask_index=None,
        noise_disable_threshold=None,
        text_uncond_prob=None,
        x_noisy_ref=None,
    ):
        """
        Compute training losses for a single timestep.
        Arguments format copied from opensora/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """

        y_null = None
        if mask_index is not None and len(mask_index) > 0:  # i2v and v2v
            num_frames = x_start.shape[2]
            x_cond_mask = torch.zeros_like(x_start, device=x_start.device)
            x_cond_mask[:, :, mask_index, :, :] = 1.0
            x_noisy_ref = x_noisy_ref if x_noisy_ref is not None else x_start
            x_cond = x_noisy_ref * x_cond_mask
            y_null = (
                model.module.y_embedder.y_embedding[None]
                .repeat(model_kwargs["y"].shape[1], 1, 1)[:, None]
                .expand_as(model_kwargs["y"])
            )
            condition = random.choices(
                list(self.drop_condition.keys()), weights=list(self.drop_condition.values()), k=1
            )[0]
            if condition == "null":  # no text, no x_cond
                x_cond = torch.zeros_like(x_cond).to(x_cond.device).to(x_cond.dtype)
                model_kwargs["y"] = y_null
            elif condition == "cond":  # no text
                model_kwargs["y"] = y_null
            elif condition == "text":  # no x_cond
                x_cond = torch.zeros_like(x_cond).to(x_cond.device).to(x_cond.dtype)
        else:
            x_cond_mask = torch.zeros_like(x_start, device=x_start.device)
            x_cond = x_start * x_cond_mask

        if t is None:
            t = self.time_sampler.sample(x_start, self.num_timesteps, model_kwargs)
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        if noise_disable_threshold is not None:
            no_noise_mask = t > noise_disable_threshold
            x_start[no_noise_mask] = x_gt[no_noise_mask]

        x_t = self.add_noise(x_start, noise, t)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        terms = {}
        # model_output = model(x_t, t, **model_kwargs)
        model_output = model(
            x_t,
            t,
            cond=x_cond,
            cond_mask=x_cond_mask,
            mask_index=mask_index,
            y_null=y_null,
            text_uncond_prob=text_uncond_prob,
            **model_kwargs,
        )
        velocity_pred = model_output.chunk(2, dim=1)[0]

        res_weights = None
        if mask_index is not None and len(mask_index) > 0:
            mask_nums = len(mask_index)
            new_weight = (num_frames - mask_nums * self.x_cond_weight) / (
                num_frames - mask_nums
            )  # scale to ensure sum is same
            res_weights = (
                torch.ones_like(velocity_pred).to(velocity_pred.device).to(velocity_pred.dtype) * new_weight
            )  # set non-masked frame weight
            res_weights[:, :, mask_index, :, :] = 1 * self.x_cond_weight  # set masked frame weight

        x_target = x_gt if x_gt is not None else x_start
        if weights is None:
            loss = (
                mean_flat(res_weights * (velocity_pred - (x_target - noise)).pow(2), mask=mask)
                if res_weights is not None
                else mean_flat((velocity_pred - (x_target - noise)).pow(2), mask=mask)
            )
        else:
            weight = _extract_into_tensor(weights, t, x_target.shape)
            loss = (
                mean_flat(res_weights * weight * (velocity_pred - (x_target - noise)).pow(2), mask=mask)
                if res_weights is not None
                else mean_flat(weight * (velocity_pred - (x_target - noise)).pow(2), mask=mask)
            )
        terms["loss"] = loss

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
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])

        return timepoints * original_samples + (1 - timepoints) * noise

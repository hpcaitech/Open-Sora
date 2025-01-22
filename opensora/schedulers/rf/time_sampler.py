import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import LogisticNormal


def extract_hwt(model_kwargs, scale_temporal=True):
    # Force fp16 input to fp32 to avoid nan output
    for key in ["height", "width", "num_frames"]:
        assert key in model_kwargs, f"model_kwargs must contain key {key}"
        if isinstance(model_kwargs[key], torch.Tensor) and model_kwargs[key].dtype == torch.float16:
            model_kwargs[key] = model_kwargs[key].float()

    resolution = model_kwargs["height"] * model_kwargs["width"]
    # NOTE: currently, we do not take fps into account
    # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
    if model_kwargs["num_frames"][0] == 1 or not scale_temporal:
        num_frames = torch.ones_like(model_kwargs["num_frames"])
        is_image = True
    else:
        # num_frames = model_kwargs["num_frames"] // 17 * 5
        num_frames = (model_kwargs["num_frames"] - 1) // 4
        is_image = False
    return resolution, num_frames, is_image


def timestep_transform(
    t,
    model_kwargs,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
    scale_temporal=True,
    ret_ratio=False,
    uniform_over_threshold=None,
):
    t = t / num_timesteps

    resolution, num_frames, is_image = extract_hwt(model_kwargs, scale_temporal)

    ratio_space = (resolution / base_resolution).sqrt()
    ratio_time = (num_frames / base_num_frames).sqrt()
    ratio = ratio_space * ratio_time
    if not is_image:
        ratio = ratio * scale
    t = ratio * t / (1 + (ratio - 1) * t)

    if uniform_over_threshold is not None and not is_image:
        t_over = t > uniform_over_threshold
        t_resample = torch.rand_like(t) * (1 - uniform_over_threshold) + uniform_over_threshold
        t = t_over * t_resample + ~t_over * t

    t = t * num_timesteps
    if ret_ratio:
        return t, ratio
    return t


class TimeSampler:
    def __init__(
        self,
        sample_method="uniform",
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        transform_scale=1.0,
        scale_temporal=True,
        loc=0.0,
        scale=1.0,
        uniform_over_threshold=None,
    ):
        assert sample_method in ["uniform", "logit-normal"]
        assert (
            sample_method == "uniform" or not use_discrete_timesteps
        ), "Only uniform sampling is supported for discrete timesteps"

        self.sample_method = sample_method
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale
        self.scale_temporal = scale_temporal
        self.uniform_over_threshold = uniform_over_threshold
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

    def sample(self, x_start, num_timesteps, model_kwargs=None):
        if self.use_discrete_timesteps:
            t = torch.randint(0, num_timesteps, (x_start.shape[0],), device=x_start.device)
        elif self.sample_method == "uniform":
            t = torch.rand((x_start.shape[0],), device=x_start.device) * num_timesteps
        elif self.sample_method == "logit-normal":
            t = self.sample_t(x_start) * num_timesteps

        if not self.use_timestep_transform:
            return t

        t = timestep_transform(
            t,
            model_kwargs,
            scale=self.transform_scale,
            num_timesteps=num_timesteps,
            scale_temporal=self.scale_temporal,
            uniform_over_threshold=self.uniform_over_threshold,
        )

        return t

    def visualize(self, height=360, width=640, num_frames=113, num_timesteps=1000):
        bs = 10000
        x_start = torch.randn(bs)
        self.use_timestep_transform = False
        original_t_values = self.sample(x_start, num_timesteps)
        self.use_timestep_transform = True

        color = "#8E7CC3"
        label = f"({height}x{width})"
        bins = np.linspace(0, 1000, 1000)
        fig, axes = plt.subplots(2, 1, figsize=(18, 18))

        model_kwargs = {
            "height": torch.full((bs,), height),
            "width": torch.full((bs,), width),
            "num_frames": torch.full((bs,), num_frames),
        }
        transformed_t_values, ratio = timestep_transform(
            original_t_values,
            model_kwargs,
            scale=2,
            num_timesteps=num_timesteps,
            scale_temporal=self.scale_temporal,
            ret_ratio=True,
            uniform_over_threshold=0.95,
        )

        # Scatter with lines
        axes[0].scatter(
            original_t_values.numpy(),
            transformed_t_values.numpy(),
            label=f"{label} (scale={ratio[0].item()})",
            s=10,
            color=color,
        )
        for j in range(len(original_t_values)):
            axes[0].plot(
                [original_t_values[j], original_t_values[j]],
                [original_t_values[j], transformed_t_values[j]],
                color="gray",
                alpha=0.5,
            )

        axes[0].scatter(
            original_t_values.numpy(), original_t_values.numpy(), label="Original $t$", color="#C27BA0", s=10
        )  # Soft Rose
        axes[0].set_xlabel("Original $t$")
        axes[0].set_ylabel("Transformed $new_t$")
        axes[0].set_title(f"{label} Transformation")
        axes[0].legend()
        axes[0].grid(True)

        # Histogram
        axes[1].hist(original_t_values.numpy(), bins=bins, alpha=0.6, label="Original", color="#C27BA0")
        axes[1].hist(transformed_t_values.numpy(), bins=bins, alpha=0.6, label=f"Transformed ({label})", color=color)

        axes[1].set_xlabel("Adjusted $new_t$")
        axes[1].set_ylabel("Number of Points")
        axes[1].set_title(f"Distribution of Original and Transformed $t$ ({label})")
        axes[1].legend()

        plt.tight_layout()
        path = "./samples/infos/timestep_sampling_comparison.png"
        plt.savefig(path)
        print(f"Saved figure to {path}")

        prob_80 = ((transformed_t_values > 800).sum() / len(transformed_t_values)).item()
        prob_85 = ((transformed_t_values > 850).sum() / len(transformed_t_values)).item()
        prob_90 = ((transformed_t_values > 900).sum() / len(transformed_t_values)).item()
        prob_95 = ((transformed_t_values > 950).sum() / len(transformed_t_values)).item()
        prob_99 = ((transformed_t_values > 990).sum() / len(transformed_t_values)).item()
        print(f"Ratio: {ratio[0].item()}")
        print(f"Probability of t > 800: {prob_80}")
        print(f"Probability of t > 850: {prob_85}")
        print(f"Probability of t > 900: {prob_90}")
        print(f"Probability of t > 950: {prob_95}")
        print(f"Probability of t > 990: {prob_99}")


if __name__ == "__main__":
    time_sampler = TimeSampler(
        use_timestep_transform=True,
        sample_method="logit-normal",
    )
    time_sampler.visualize()

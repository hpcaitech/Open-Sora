import math
import random
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from colossalai.booster.plugin import LowLevelZeroPlugin
from einops import rearrange

from opensora.acceleration.parallel_states import set_data_parallel_group, set_sequence_parallel_group
from opensora.acceleration.plugin import ZeroSeqParallelPlugin

from .misc import get_logger


def create_colossalai_plugin(plugin, dtype, grad_clip, sp_size, reduce_bucket_size_in_m: int = 20):
    plugin_kwargs = dict(
        precision=dtype,
        initial_scale=2**16,
        max_norm=grad_clip,
        reduce_bucket_size_in_m=reduce_bucket_size_in_m,
        overlap_allgather=True,
        cast_inputs=False,
    )
    if plugin == "zero1" or plugin == "zero2":
        assert sp_size == 1, "Zero plugin does not support sequence parallelism"
        stage = 1 if plugin == "zero1" else 2
        plugin = LowLevelZeroPlugin(
            stage=stage,
            **plugin_kwargs,
        )
        set_data_parallel_group(dist.group.WORLD)
    elif plugin == "zero1-seq" or plugin == "zero2-seq":
        assert sp_size > 1, "Zero-seq plugin requires sequence parallelism"
        stage = 1 if plugin == "zero1-seq" else 2
        plugin = ZeroSeqParallelPlugin(
            sp_size=sp_size,
            stage=stage,
            **plugin_kwargs,
        )
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
    else:
        raise ValueError(f"Unknown plugin {plugin}")
    return plugin


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, optimizer=None, decay: float = 0.9999, sharded: bool = True
) -> None:
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == "pos_embed":
            continue
        if not param.requires_grad:
            continue
        if not sharded:
            param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
        else:
            if param.data.dtype != torch.float32:
                param_id = id(param)
                master_param = optimizer.get_working_to_master_map()[param_id]
                param_data = master_param.data
            else:
                param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)


class MaskGenerator:
    def __init__(self, mask_ratios):
        valid_mask_names = [
            "identity",
            "quarter_random",
            "quarter_head",
            "quarter_tail",
            "quarter_head_tail",
            "image_random",
            "image_head",
            "image_tail",
            "image_head_tail",
            "random",
            "intepolate",
        ]
        assert all(
            mask_name in valid_mask_names for mask_name in mask_ratios.keys()
        ), f"mask_name should be one of {valid_mask_names}, got {mask_ratios.keys()}"
        assert all(
            mask_ratio >= 0 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be greater than or equal to 0, got {mask_ratios.values()}"
        assert all(
            mask_ratio <= 1 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be less than or equal to 1, got {mask_ratios.values()}"
        # sum of mask_ratios should be 1
        if "identity" not in mask_ratios:
            mask_ratios["identity"] = 1.0 - sum(mask_ratios.values())
        assert math.isclose(
            sum(mask_ratios.values()), 1.0, abs_tol=1e-6
        ), f"sum of mask_ratios should be 1, got {sum(mask_ratios.values())}"
        get_logger().info("mask ratios: %s", mask_ratios)
        self.mask_ratios = mask_ratios

    def get_mask(self, x):
        mask_type = random.random()
        mask_name = None
        prob_acc = 0.0
        for mask, mask_ratio in self.mask_ratios.items():
            prob_acc += mask_ratio
            if mask_type < prob_acc:
                mask_name = mask
                break

        num_frames = x.shape[2]
        # Hardcoded condition_frames
        condition_frames_max = num_frames // 4

        mask = torch.ones(num_frames, dtype=torch.bool, device=x.device)
        if num_frames <= 1:
            return mask

        if mask_name == "quarter_random":
            random_size = random.randint(1, condition_frames_max)
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "image_random":
            random_size = 1
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "quarter_head":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
        elif mask_name == "image_head":
            random_size = 1
            mask[:random_size] = 0
        elif mask_name == "quarter_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[-random_size:] = 0
        elif mask_name == "image_tail":
            random_size = 1
            mask[-random_size:] = 0
        elif mask_name == "quarter_head_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "image_head_tail":
            random_size = 1
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "intepolate":
            random_start = random.randint(0, 1)
            mask[random_start::2] = 0
        elif mask_name == "random":
            mask_ratio = random.uniform(0.1, 0.9)
            mask = torch.rand(num_frames, device=x.device) > mask_ratio
        # if mask is all False, set the last frame to True
        if not mask.any():
            mask[-1] = 1

        return mask

    def get_masks(self, x):
        masks = []
        for _ in range(len(x)):
            mask = self.get_mask(x)
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        return masks


def class_dropout(text_list, drop_ratio):
    # replace text with "" in text_list with probability drop_ratio
    text_ret = []
    for i in range(len(text_list)):
        if random.random() < drop_ratio:
            text_ret.append("")
        else:
            text_ret.append(text_list[i])
    return text_ret


# Noise augmentation


def add_noise(x, noise_level: float):
    noise = torch.randn_like(x)
    x_noise = (1 - noise_level) * x + noise_level * noise
    return x_noise


def downsample_spatial(x, downsample_ratio: int):
    assert downsample_ratio in [1, 2, 4, 8], f"downsample_ratio should be one of [1, 2, 4, 8], got {downsample_ratio}"
    bs = x.shape[0]
    x = rearrange(x, "b c t h w -> (b t) c h w")

    downsampled_image = F.interpolate(x, scale_factor=1 / downsample_ratio, mode="bilinear", align_corners=False)
    upsampled_image = F.interpolate(downsampled_image, size=x.shape[2:], mode="bilinear", align_corners=False)

    x = rearrange(upsampled_image, "(b t) c h w -> b c t h w", b=bs)

    return x


def downsample_temporal(x, downsample_ratio: int):
    assert downsample_ratio in [1, 2, 4, 8], f"downsample_ratio should be one of [1, 2, 4, 8], got {downsample_ratio}"

    downsampled_video = F.interpolate(
        x,
        scale_factor=(1 / downsample_ratio, 1, 1),
        mode="trilinear",
        align_corners=False,
    )
    upsampled_video = F.interpolate(
        downsampled_video,
        size=x.shape[2:],
        mode="trilinear",
        align_corners=False,
    )

    return upsampled_video


# noise_injection_prob = cfg.get("noise_injection_prob", 0.0)
# if noise_injection_prob > 0 and random.random() < noise_injection_prob:
# noise_upper_bound = cfg.get("noise_upper_bound", 0.3)


def aug_x(x, vae, prob_dict, strength_dict):
    x_gt = vae.encode(x)

    # downsample_spatial
    if random.random() < prob_dict.get("downsample_spatial", 0.0):
        st = random.choice(strength_dict["downsample_spatial"])
        x = downsample_spatial(x, st)

    # downsample_temporal
    if random.random() < prob_dict.get("downsample_temporal", 0.0):
        st = random.choice(strength_dict["downsample_temporal"])
        x = downsample_temporal(x, st)

    # gaussian_pixel
    if random.random() < prob_dict.get("gaussian_pixel", 0.0):
        st = random.random() * strength_dict["gaussian_pixel"]
        x = add_noise(x, st)

    x = vae.encode(x)

    # gaussian_feature
    if random.random() < prob_dict.get("gaussian_feature", 0.0):
        st = random.random() * strength_dict["gaussian_feature"]
        x = add_noise(x, st)

    return x, x_gt


def get_mask_cond(randgen, mask_types) -> str:
    mask_cond = randgen.choices(list(mask_types.keys()), weights=list(mask_types.values()), k=1)[0]
    return mask_cond


def get_mask_index(mask_cond, latent_t):
    if mask_cond == "v2v_head" or mask_cond == "v2v_head_noisy":
        mask_index = [k for k in range(latent_t // 2)]
    elif mask_cond == "v2v_tail":
        mask_index = [k for k in range(latent_t // 2, latent_t)]
    elif mask_cond == "i2v" or mask_cond == "i2v_head":  # equivalent
        mask_index = [0]
    elif mask_cond == "i2v_loop":
        mask_index = [0, latent_t - 1]
    elif mask_cond == "i2v_tail":
        mask_index = [latent_t - 1]
    elif mask_cond == "other" or mask_cond == "other_noisy":
        edge = random.choices([0, latent_t - 1], k=1)[0]
        if edge == 0:
            mask_index = [
                k for k in range(random.randint(1, latent_t - 2))
            ]  # IMPORTANT: don't allow full mask as not useful and x_pred will have zero T size
        else:
            mask_index = [k for k in range(random.randint(1, latent_t - 2), latent_t)]
    elif mask_cond == "none":
        mask_index = []
    else:
        raise NotImplementedError

    return mask_index

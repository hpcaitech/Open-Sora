import random
import warnings
from collections import OrderedDict
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from colossalai.booster.plugin import HybridParallelPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.utils import get_current_device
from einops import rearrange
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from opensora.acceleration.parallel_states import (
    set_data_parallel_group,
    set_sequence_parallel_group,
    set_tensor_parallel_group,
)
from opensora.utils.optimizer import LinearWarmupLR


def set_lr(
    optimizer: torch.optim.Optimizer,
    lr_scheduler: _LRScheduler,
    lr: float,
    initial_lr: float = None,
):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    if isinstance(lr_scheduler, LinearWarmupLR):
        lr_scheduler.base_lrs = [lr] * len(lr_scheduler.base_lrs)
        if initial_lr is not None:
            lr_scheduler.initial_lr = initial_lr


def set_warmup_steps(
    lr_scheduler: _LRScheduler,
    warmup_steps: int,
):
    if isinstance(lr_scheduler, LinearWarmupLR):
        lr_scheduler.warmup_steps = warmup_steps


def set_eps(
    optimizer: torch.optim.Optimizer,
    eps: float = None,
):
    if eps is not None:
        for param_group in optimizer.param_groups:
            param_group["eps"] = eps


def setup_device() -> tuple[torch.device, DistCoordinator]:
    """
    Setup the device and the distributed coordinator.

    Returns:
        tuple[torch.device, DistCoordinator]: The device and the distributed coordinator.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # NOTE: A very large timeout is set to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    coordinator = DistCoordinator()
    device = get_current_device()

    return device, coordinator


def create_colossalai_plugin(
    plugin: str,
    dtype: str,
    grad_clip: float,
    **kwargs,
) -> LowLevelZeroPlugin | HybridParallelPlugin:
    """
    Create a ColossalAI plugin.

    Args:
        plugin (str): The plugin name.
        dtype (str): The data type.
        grad_clip (float): The gradient clip value.

    Returns:
        LowLevelZeroPlugin |  HybridParallelPlugin: The plugin.
    """
    plugin_kwargs = dict(
        precision=dtype,
        initial_scale=2**16,
        max_norm=grad_clip,
        overlap_allgather=True,
        cast_inputs=False,
        reduce_bucket_size_in_m=20,
    )
    plugin_kwargs.update(kwargs)
    sp_size = plugin_kwargs.get("sp_size", 1)
    if plugin == "zero1" or plugin == "zero2":
        assert sp_size == 1, "Zero plugin does not support sequence parallelism"
        stage = 1 if plugin == "zero1" else 2
        plugin = LowLevelZeroPlugin(
            stage=stage,
            **plugin_kwargs,
        )
        set_data_parallel_group(dist.group.WORLD)
    elif plugin == "hybrid":
        plugin_kwargs["find_unused_parameters"] = True
        reduce_bucket_size_in_m = plugin_kwargs.pop("reduce_bucket_size_in_m")
        if "zero_bucket_size_in_m" not in plugin_kwargs:
            plugin_kwargs["zero_bucket_size_in_m"] = reduce_bucket_size_in_m
        plugin_kwargs.pop("cast_inputs")
        plugin_kwargs["enable_metadata_cache"] = False

        custom_policy = plugin_kwargs.pop("custom_policy", None)
        if custom_policy is not None:
            custom_policy = custom_policy()
        plugin = HybridParallelPlugin(
            custom_policy=custom_policy,
            **plugin_kwargs,
        )
        set_tensor_parallel_group(plugin.tp_group)
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
    else:
        raise ValueError(f"Unknown plugin {plugin}")
    return plugin


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, optimizer=None, decay: float = 0.9999, sharded: bool = True
):
    """
    Step the EMA model towards the current model.

    Args:
        ema_model (torch.nn.Module): The EMA model.
        model (torch.nn.Module): The current model.
        optimizer (torch.optim.Optimizer): The optimizer.
        decay (float): The decay rate.
        sharded (bool): Whether the model is sharded.
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


def dropout_condition(prob: float, txt: torch.Tensor, null_txt: torch.Tensor) -> torch.Tensor:
    """
    Apply dropout to the text tensor.

    Args:
        prob (float): The dropout probability.
        txt (torch.Tensor): The text tensor.
        null_txt (torch.Tensor): The null text tensor.

    Returns:
        torch.Tensor: The text tensor with dropout applied.
    """
    if prob == 0:
        warnings.warn("Dropout probability is 0, skipping dropout")
    drop_ids = torch.rand(txt.shape[0], device=txt.device) < prob
    drop_ids = drop_ids.view((drop_ids.shape[0],) + (1,) * (txt.ndim - 1))
    new_txt = torch.where(drop_ids, null_txt, txt)
    return new_txt


def prepare_visual_condition_uncausal(
    x: torch.Tensor, condition_config: dict, model_ae: torch.nn.Module, pad: bool = False
) -> torch.Tensor:
    """
    Prepare the visual condition for the model.

    Args:
        x: (torch.Tensor): The input video tensor.
        condition_config (dict): The condition configuration.
        model_ae (torch.nn.Module): The video encoder module.

    Returns:
        torch.Tensor: The visual condition tensor.
    """
    # x has shape [b, c, t, h, w], where b is the batch size
    B = x.shape[0]
    C = model_ae.cfg.latent_channels
    T, H, W = model_ae.get_latent_size(x.shape[-3:])

    # Initialize masks tensor to match the shape of x, but only the time dimension will be masked
    masks = torch.zeros(B, 1, T, H, W).to(
        x.device, x.dtype
    )  # broadcasting over channel, concat to masked_x with 1 + 16 = 17 channesl
    # to prevent information leakage, image must be encoded separately and copied to latent
    latent = torch.zeros(B, C, T, H, W).to(x.device, x.dtype)
    x_0 = torch.zeros(B, C, T, H, W).to(x.device, x.dtype)
    if T > 1:  # video
        # certain v2v conditions not are applicable for short videos
        if T <= 32 // model_ae.time_compression_ratio:
            condition_config.pop("v2v_head", None)  # given first 32 frames
            condition_config.pop("v2v_tail", None)  # given last 32 frames
            condition_config.pop("v2v_head_easy", None)  # given first 64 frames
            condition_config.pop("v2v_tail_easy", None)  # given last 64 frames
        if T <= 64 // model_ae.time_compression_ratio:
            condition_config.pop("v2v_head_easy", None)  # given first 64 frames
            condition_config.pop("v2v_tail_easy", None)  # given last 64 frames

        mask_cond_options = list(condition_config.keys())  # list of mask conditions
        mask_cond_weights = list(condition_config.values())  # corresponding probabilities

        for i in range(B):
            # Randomly select a mask condition based on the provided probabilities
            mask_cond = random.choices(mask_cond_options, weights=mask_cond_weights, k=1)[0]
            # Apply the selected mask condition directly on the masks tensor
            if mask_cond == "i2v_head":  # NOTE: modify video, mask first latent frame
                # padded video such that the first latent frame correspond to image only
                masks[i, :, 0, :, :] = 1
                if pad:
                    pad_num = model_ae.time_compression_ratio - 1  # 32 --> new video: 7 + (1+31-7)
                    padded_x = torch.cat([x[i, :, :1]] * pad_num + [x[i, :, :-pad_num]], dim=1).unsqueeze(0)
                    x_0[i] = model_ae.encode(padded_x)[0]
                else:
                    x_0[i] = model_ae.encode(x[i : i + 1])[0]
                # condition: encode the image only
                latent[i, :, :1, :, :] = model_ae.encode(
                    x[i, :, :1, :, :].unsqueeze(0)
                )  # since the first dimension of right hand side is singleton, torch auto-ignores it
            elif mask_cond == "i2v_loop":  # # NOTE: modify video, mask first and last latent frame
                # pad video such that first and last latent frame correspond to image only
                masks[i, :, 0, :, :] = 1
                masks[i, :, -1, :, :] = 1
                if pad:
                    pad_num = model_ae.time_compression_ratio - 1
                    padded_x = torch.cat(
                        [x[i, :, :1]] * pad_num
                        + [x[i, :, : -pad_num * 2]]
                        + [x[i, :, -pad_num * 2 - 1].unsqueeze(1)] * pad_num,
                        dim=1,
                    ).unsqueeze(
                        0
                    )  # remove the last pad_num * 2 frames from the end of the video
                    x_0[i] = model_ae.encode(padded_x)[0]
                    # condition: encode the image only
                    latent[i, :, :1, :, :] = model_ae.encode(x[i, :, :1, :, :].unsqueeze(0))
                    latent[i, :, -1:, :, :] = model_ae.encode(x[i, :, -pad_num * 2 - 1, :, :].unsqueeze(1).unsqueeze(0))
                else:
                    x_0[i] = model_ae.encode(x[i : i + 1])[0]
                    latent[i, :, :1, :, :] = model_ae.encode(x[i, :, :1, :, :].unsqueeze(0))
                    latent[i, :, -1:, :, :] = model_ae.encode(x[i, :, -1:, :, :].unsqueeze(0))
            elif mask_cond == "i2v_tail":  # mask the last latent frame
                masks[i, :, -1, :, :] = 1
                if pad:
                    pad_num = model_ae.time_compression_ratio - 1
                    padded_x = torch.cat([x[i, :, pad_num:]] + [x[i, :, -1:]] * pad_num, dim=1).unsqueeze(0)
                    x_0[i] = model_ae.encode(padded_x)[0]
                    latent[i, :, -1:, :, :] = model_ae.encode(x[i, :, -pad_num * 2 - 1, :, :].unsqueeze(1).unsqueeze(0))
                else:
                    x_0[i] = model_ae.encode(x[i : i + 1])[0]
                    latent[i, :, -1:, :, :] = model_ae.encode(x[i, :, -1:, :, :].unsqueeze(0))
            elif mask_cond == "v2v_head":  # mask the first 32 video frames
                assert T > 32 // model_ae.time_compression_ratio
                conditioned_t = 32 // model_ae.time_compression_ratio
                masks[i, :, :conditioned_t, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, :conditioned_t, :, :] = x_0[i, :, :conditioned_t, :, :]
            elif mask_cond == "v2v_tail":  # mask the last 32 video frames
                assert T > 32 // model_ae.time_compression_ratio
                conditioned_t = 32 // model_ae.time_compression_ratio
                masks[i, :, -conditioned_t:, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, -conditioned_t:, :, :] = x_0[i, :, -conditioned_t:, :, :]
            elif mask_cond == "v2v_head_easy":  # mask the first 64 video frames
                assert T > 64 // model_ae.time_compression_ratio
                conditioned_t = 64 // model_ae.time_compression_ratio
                masks[i, :, :conditioned_t, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, :conditioned_t, :, :] = x_0[i, :, :conditioned_t, :, :]
            elif mask_cond == "v2v_tail_easy":  # mask the last 64 video frames
                assert T > 64 // model_ae.time_compression_ratio
                conditioned_t = 64 // model_ae.time_compression_ratio
                masks[i, :, -conditioned_t:, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, -conditioned_t:, :, :] = x_0[i, :, -conditioned_t:, :, :]
            # elif mask_cond == "v2v_head":  # mask from the beginning to a random point
            #     masks[i, :, : random.randint(1, T - 2), :, :] = 1
            # elif mask_cond == "v2v_tail":  # mask from a random point to the end
            #     masks[i, :, -random.randint(1, T - 2) :, :, :] = 1
            else:
                # "t2v" is the fallback case where no specific condition is specified
                assert mask_cond == "t2v", f"Unknown mask condition {mask_cond}"
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
    else:  # image
        x_0 = model_ae.encode(x)  # latent video

    latent = masks * latent  # condition latent
    # merge the masks and the masked_x into a single tensor
    cond = torch.cat((masks, latent), dim=1)
    return x_0, cond


def prepare_visual_condition_causal(x: torch.Tensor, condition_config: dict, model_ae: torch.nn.Module) -> torch.Tensor:
    """
    Prepare the visual condition for the model.

    Args:
        x: (torch.Tensor): The input video tensor.
        condition_config (dict): The condition configuration.
        model_ae (torch.nn.Module): The video encoder module.

    Returns:
        torch.Tensor: The visual condition tensor.
    """
    # x has shape [b, c, t, h, w], where b is the batch size
    B = x.shape[0]
    C = model_ae.cfg.latent_channels
    T, H, W = model_ae.get_latent_size(x.shape[-3:])

    # Initialize masks tensor to match the shape of x, but only the time dimension will be masked
    masks = torch.zeros(B, 1, T, H, W).to(
        x.device, x.dtype
    )  # broadcasting over channel, concat to masked_x with 1 + 16 = 17 channesl
    # to prevent information leakage, image must be encoded separately and copied to latent
    latent = torch.zeros(B, C, T, H, W).to(x.device, x.dtype)
    x_0 = torch.zeros(B, C, T, H, W).to(x.device, x.dtype)
    if T > 1:  # video
        # certain v2v conditions not are applicable for short videos
        if T <= (32 // model_ae.time_compression_ratio) + 1:
            condition_config.pop("v2v_head", None)  # given first 33 frames
            condition_config.pop("v2v_tail", None)  # given last 33 frames
            condition_config.pop("v2v_head_easy", None)  # given first 65 frames
            condition_config.pop("v2v_tail_easy", None)  # given last 65 frames
        if T <= (64 // model_ae.time_compression_ratio) + 1:
            condition_config.pop("v2v_head_easy", None)  # given first 65 frames
            condition_config.pop("v2v_tail_easy", None)  # given last 65 frames

        mask_cond_options = list(condition_config.keys())  # list of mask conditions
        mask_cond_weights = list(condition_config.values())  # corresponding probabilities

        for i in range(B):
            # Randomly select a mask condition based on the provided probabilities
            mask_cond = random.choices(mask_cond_options, weights=mask_cond_weights, k=1)[0]
            # Apply the selected mask condition directly on the masks tensor

            if mask_cond == "i2v_head":  # NOTE: modify video, mask first latent frame
                masks[i, :, 0, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                # condition: encode the image only
                latent[i, :, :1, :, :] = model_ae.encode(x[i, :, :1, :, :].unsqueeze(0))

            elif mask_cond == "i2v_loop":  # # NOTE: modify video, mask first and last latent frame
                # pad video such that first and last latent frame correspond to image only
                masks[i, :, 0, :, :] = 1
                masks[i, :, -1, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                # condition: encode the image only
                latent[i, :, :1, :, :] = model_ae.encode(x[i, :, :1, :, :].unsqueeze(0))
                latent[i, :, -1:, :, :] = model_ae.encode(x[i, :, -1:, :, :].unsqueeze(0))

            elif mask_cond == "i2v_tail":  # mask the last latent frame
                masks[i, :, -1, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                # condition: encode the last image only
                latent[i, :, -1:, :, :] = model_ae.encode(x[i, :, -1:, :, :].unsqueeze(0))

            elif "v2v_head" in mask_cond:  # mask the first 33 video frames
                ref_t = 33 if not "easy" in mask_cond else 65
                assert (ref_t - 1) % model_ae.time_compression_ratio == 0
                conditioned_t = (ref_t - 1) // model_ae.time_compression_ratio + 1
                masks[i, :, :conditioned_t, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                # encode the first ref_t frame video separately
                latent[i, :, :conditioned_t, :, :] = model_ae.encode(x[i, :, :ref_t, :, :].unsqueeze(0))

            elif "v2v_tail" in mask_cond:  # mask the last 32 video frames
                ref_t = 33 if not "easy" in mask_cond else 65
                assert (ref_t - 1) % model_ae.time_compression_ratio == 0
                conditioned_t = (ref_t - 1) // model_ae.time_compression_ratio + 1
                masks[i, :, -conditioned_t:, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                # encode the first ref_t frame video separately
                latent[i, :, -conditioned_t:, :, :] = model_ae.encode(x[i, :, -ref_t:, :, :].unsqueeze(0))
            else:
                # "t2v" is the fallback case where no specific condition is specified
                assert mask_cond == "t2v", f"Unknown mask condition {mask_cond}"
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
    else:  # image
        x_0 = model_ae.encode(x)  # latent video

    latent = masks * latent  # condition latent
    # merge the masks and the masked_x into a single tensor
    cond = torch.cat((masks, latent), dim=1)
    return x_0, cond


def get_batch_loss(model_pred, v_t, masks=None):
    # for I2V, only include the generated frames in loss calculation
    if masks is not None:  # shape [B, T, H, W]
        num_frames, height, width = masks.shape[-3:]
        masks = masks[:, :, 0, 0]  # only look at [B, T]
        model_pred = rearrange(
            model_pred,
            "b (t h w) (c ph pw) -> b c t (h ph) (w pw)",
            h=height // 2,
            w=width // 2,
            t=num_frames,
            ph=2,
            pw=2,
        )
        v_t = rearrange(
            v_t,
            "b (t h w) (c ph pw) -> b c t (h ph) (w pw)",
            h=height // 2,
            w=width // 2,
            t=num_frames,
            ph=2,
            pw=2,
        )

        batch_loss = 0
        for i in range(model_pred.size(0)):
            pred_val = model_pred[i]
            target_val = v_t[i]
            if masks[i][0] == 1 and (not 1 in masks[i][1:-1]):  # have front padding
                pred_val = pred_val[:, 1:]
                target_val = target_val[:, 1:]
            if masks[i][-1] == 1 and (not 1 in masks[i][1:-1]):  # have tail padding
                pred_val = pred_val[:, :-1]
                target_val = target_val[:, :-1]
            batch_loss += F.mse_loss(pred_val.float(), target_val.float(), reduction="mean")
            # print(f"mask {masks[i]}, pred_val shape: {pred_val.size()}")
        loss = batch_loss / model_pred.size(0)
    else:
        # use reduction mean so that each batch will have same level of influence regardless of batch size
        loss = F.mse_loss(model_pred.float(), v_t.float(), reduction="mean")
    return loss


@torch.no_grad()
def warmup_ae(model_ae: nn.Module, shapes: list[tuple[int, ...]], device: torch.device, dtype: torch.dtype):
    progress_bar = tqdm(shapes, desc="Warmup AE", disable=dist.get_rank() != 0)
    for x_shape in progress_bar:
        x = torch.randn(*x_shape, device=device, dtype=dtype)
        _ = model_ae.encode(x)

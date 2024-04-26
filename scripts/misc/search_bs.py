import argparse
import time
import traceback
from copy import deepcopy

import colossalai
import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from mmengine.config import Config
from tqdm import tqdm

from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)
from opensora.acceleration.plugin import ZeroSeqParallelPlugin
from opensora.datasets import prepare_variable_dataloader
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import model_sharding
from opensora.utils.config_utils import merge_args, parse_configs
from opensora.utils.misc import format_numel_str, get_model_numel, requires_grad, to_torch_dtype
from opensora.utils.train_utils import MaskGenerator, update_ema


class BColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# BUCKETS = [
#     ("240p", 16),
#     ("240p", 32),
#     ("240p", 64),
#     ("240p", 128),
#     ("256", 1),
#     ("512", 1),
#     ("480p", 1),
#     ("480p", 16),
#     ("480p", 32),
#     ("720p", 16),
#     ("720p", 32),
#     ("1024", 1),
#     ("1080p", 1),
# ]


def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="model config file path")
    parser.add_argument("-o", "--output", help="output config file path", default="output_config.py")

    parser.add_argument("--seed", default=42, type=int, help="generation seed")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        help="path to model ckpt; will overwrite cfg.ckpt_path if specified",
    )
    parser.add_argument("--data-path", default=None, type=str, help="path to data csv", required=True)
    parser.add_argument("--warmup-steps", default=1, type=int, help="warmup steps")
    parser.add_argument("--active-steps", default=1, type=int, help="active steps")
    parser.add_argument("--base-resolution", default="240p", type=str, help="base resolution")
    parser.add_argument("--base-frames", default=128, type=int, help="base frames")
    parser.add_argument("--batch-size-start", default=2, type=int, help="batch size start")
    parser.add_argument("--batch-size-end", default=256, type=int, help="batch size end")
    parser.add_argument("--batch-size-step", default=2, type=int, help="batch size step")
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args, training=True)
    return cfg, args


def rewrite_config(cfg, resolution, num_frames, batch_size):
    cfg.bucket_config = {resolution: {num_frames: (1.0, batch_size)}}
    return cfg


def update_bucket_config_bs(bucket_config, resolution, num_frames, batch_size):
    p, _ = bucket_config[resolution][num_frames]
    bucket_config[resolution][num_frames] = (p, batch_size)


def main():
    # ======================================================
    # 1. args & cfg
    # ======================================================
    cfg, args = parse_configs()
    print(cfg)
    assert cfg.dataset.type == "VariableVideoTextDataset", "Only VariableVideoTextDataset is supported"

    # ======================================================
    # 2. runtime variables & colossalai launch
    # ======================================================
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert cfg.dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg.dtype}"

    # 2.1. colossalai init distributed training
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()
    device = get_current_device()
    dtype = to_torch_dtype(cfg.dtype)

    # 2.3. initialize ColossalAI booster
    if cfg.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
        )
        set_data_parallel_group(dist.group.WORLD)
    elif cfg.plugin == "zero2-seq":
        plugin = ZeroSeqParallelPlugin(
            sp_size=cfg.sp_size,
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
        )
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
    else:
        raise ValueError(f"Unknown plugin {cfg.plugin}")
    booster = Booster(plugin=plugin)

    # ======================================================
    # 4. build model
    # ======================================================
    # 4.1. build model
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS)
    input_size = (cfg.dataset.num_frames, *cfg.dataset.image_size)
    latent_size = vae.get_latent_size(input_size)
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        dtype=dtype,
    )
    model_numel, model_numel_trainable = get_model_numel(model)
    coordinator.print_on_master(
        f"Trainable model params: {format_numel_str(model_numel_trainable)}, Total model params: {format_numel_str(model_numel)}"
    )

    # 4.2. create ema
    ema = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)

    # 4.3. move to device
    vae = vae.to(device, dtype)
    model = model.to(device, dtype)

    # 4.4. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # 4.5. setup optimizer
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=0,
        adamw_mode=True,
    )
    lr_scheduler = None

    # 4.6. prepare for training
    if cfg.grad_checkpoint:
        set_grad_checkpoint(model)
    model.train()
    update_ema(ema, model, decay=0, sharded=False)
    ema.eval()
    if cfg.mask_ratios is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)
    else:
        mask_generator = None

    # =======================================================
    # 5. boost model for distributed training with colossalai
    # =======================================================
    torch.set_default_dtype(dtype)
    model, optimizer, _, _, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    torch.set_default_dtype(torch.float)
    coordinator.print_on_master("Boost model for distributed training")

    model_sharding(ema)

    buckets = [
        (res, f) for res, d in cfg.bucket_config.items() for f, (p, bs) in d.items() if bs is not None and p > 0.0
    ]
    output_bucket_cfg = deepcopy(cfg.bucket_config)
    # find the base batch size
    assert (args.base_resolution, args.base_frames) in buckets
    del buckets[buckets.index((args.base_resolution, args.base_frames))]
    base_batch_size, base_step_time = benchmark(
        args,
        cfg,
        args.base_resolution,
        args.base_frames,
        device,
        dtype,
        booster,
        vae,
        text_encoder,
        model,
        mask_generator,
        scheduler,
        optimizer,
        ema,
    )
    update_bucket_config_bs(output_bucket_cfg, args.base_resolution, args.base_frames, base_batch_size)
    coordinator.print_on_master(
        f"{BColors.OKBLUE}Base resolution: {args.base_resolution}, Base frames: {args.base_frames}, Batch size: {base_batch_size}, Base step time: {base_step_time}{BColors.ENDC}"
    )
    result_table = [f"{args.base_resolution}, {args.base_frames}, {base_batch_size}, {base_step_time:.2f}"]
    for resolution, frames in buckets:
        try:
            batch_size, step_time = benchmark(
                args,
                cfg,
                resolution,
                frames,
                device,
                dtype,
                booster,
                vae,
                text_encoder,
                model,
                mask_generator,
                scheduler,
                optimizer,
                ema,
                target_step_time=base_step_time,
            )
            coordinator.print_on_master(
                f"{BColors.OKBLUE}Resolution: {resolution}, Frames: {frames}, Batch size: {batch_size}, Step time: {step_time}{BColors.ENDC}"
            )
            update_bucket_config_bs(output_bucket_cfg, resolution, frames, batch_size)
            result_table.append(f"{resolution}, {frames}, {batch_size}, {step_time:.2f}")
        except RuntimeError:
            pass
    result_table = "\n".join(result_table)
    coordinator.print_on_master(
        f"{BColors.OKBLUE}Resolution, Frames, Batch size, Step time\n{result_table}{BColors.ENDC}"
    )
    coordinator.print_on_master(f"{BColors.OKBLUE}{output_bucket_cfg}{BColors.ENDC}")
    if coordinator.is_master():
        cfg.bucket_config = output_bucket_cfg
        cfg.dump(args.output)


def benchmark(
    args,
    cfg,
    resolution,
    num_frames,
    device,
    dtype,
    booster,
    vae,
    text_encoder,
    model,
    mask_generator,
    scheduler,
    optimizer,
    ema,
    target_step_time=None,
):
    batch_sizes = []
    step_times = []

    def run_step(bs) -> float:
        step_time = train(
            args,
            cfg,
            resolution,
            num_frames,
            bs,
            device,
            dtype,
            booster,
            vae,
            text_encoder,
            model,
            mask_generator,
            scheduler,
            optimizer,
            ema,
        )
        step_times.append(step_time)
        batch_sizes.append(bs)
        return step_time

    orig_bs = cfg.bucket_config[resolution][num_frames][1]
    lower_bound = args.batch_size_start
    upper_bound = args.batch_size_end
    step_size = args.batch_size_step
    if isinstance(orig_bs, tuple):
        if len(orig_bs) == 1:
            upper_bound = orig_bs[0]
        elif len(orig_bs) == 2:
            lower_bound, upper_bound = orig_bs
        elif len(orig_bs) == 3:
            lower_bound, upper_bound, step_size = orig_bs
    batch_start_size = lower_bound

    while lower_bound < upper_bound:
        mid = (lower_bound + upper_bound) // 2
        try:
            step_time = run_step(mid)
            lower_bound = mid + 1
        except Exception:
            traceback.print_exc()
            upper_bound = mid

    for batch_size in range(batch_start_size, upper_bound, step_size):
        if batch_size in batch_sizes:
            continue
        step_time = run_step(batch_size)
    if len(step_times) == 0:
        raise RuntimeError("No valid batch size found")
    if target_step_time is None:
        # find the fastest batch size
        throughputs = [batch_size / step_time for step_time, batch_size in zip(step_times, batch_sizes)]
        max_throughput = max(throughputs)
        target_batch_size = batch_sizes[throughputs.index(max_throughput)]
        step_time = step_times[throughputs.index(max_throughput)]
    else:
        # find the batch size that meets the target step time
        diff = [abs(t - target_step_time) for t in step_times]
        closest_step_time = min(diff)
        target_batch_size = batch_sizes[diff.index(closest_step_time)]
        step_time = step_times[diff.index(closest_step_time)]
    return target_batch_size, step_time


def train(
    args,
    cfg,
    resolution,
    num_frames,
    batch_size,
    device,
    dtype,
    booster,
    vae,
    text_encoder,
    model,
    mask_generator,
    scheduler,
    optimizer,
    ema,
):
    total_steps = args.warmup_steps + args.active_steps
    cfg = rewrite_config(deepcopy(cfg), resolution, num_frames, batch_size)

    dataset = build_module(cfg.dataset, DATASETS)
    dataset.dummy = True
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )
    dataloader = prepare_variable_dataloader(
        bucket_config=cfg.bucket_config,
        **dataloader_args,
    )
    dataloader_iter = iter(dataloader)
    num_steps_per_epoch = dataloader.batch_sampler.get_num_batch() // dist.get_world_size()

    assert num_steps_per_epoch >= total_steps, f"num_steps_per_epoch={num_steps_per_epoch} < total_steps={total_steps}"
    duration = 0
    # this is essential for the first iteration after OOM
    optimizer._grad_store.reset_all_gradients()
    optimizer._bucket_store.reset_num_elements_in_bucket()
    optimizer._bucket_store.grad_to_param_mapping = dict()
    optimizer._bucket_store._grad_in_bucket = dict()
    optimizer._bucket_store._param_list = []
    optimizer._bucket_store._padding_size = []
    for rank in range(optimizer._bucket_store._world_size):
        optimizer._bucket_store._grad_in_bucket[rank] = []
    optimizer._bucket_store.offset_list = [0]
    optimizer.zero_grad()
    for step, batch in tqdm(
        enumerate(dataloader_iter),
        desc=f"{resolution}:{num_frames} bs={batch_size}",
        total=total_steps,
    ):
        if step >= total_steps:
            break
        if step >= args.warmup_steps:
            start = time.time()

        x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
        y = batch.pop("text")
        # Visual and text encoding
        with torch.no_grad():
            # Prepare visual inputs
            x = vae.encode(x)  # [B, C, T, H/P, W/P]
            # Prepare text inputs
            model_args = text_encoder.encode(y)

        # Mask
        if cfg.mask_ratios is not None:
            mask = mask_generator.get_masks(x)
            model_args["x_mask"] = mask
        else:
            mask = None

        # Video info
        for k, v in batch.items():
            model_args[k] = v.to(device, dtype)

        # Diffusion
        t = torch.randint(0, scheduler.num_timesteps, (x.shape[0],), device=device)
        loss_dict = scheduler.training_losses(model, x, t, model_args, mask=mask)

        # Backward & update
        loss = loss_dict["loss"].mean()
        booster.backward(loss=loss, optimizer=optimizer)
        optimizer.step()
        optimizer.zero_grad()

        # Update EMA
        update_ema(ema, model.module, optimizer=optimizer)
        if step >= args.warmup_steps:
            end = time.time()
            duration += end - start

    avg_step_time = duration / args.active_steps
    return avg_step_time


if __name__ == "__main__":
    main()

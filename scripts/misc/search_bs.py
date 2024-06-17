import time
import traceback
from copy import deepcopy
from datetime import timedelta

import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm

from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.aspect import get_num_frames
from opensora.datasets.dataloader import prepare_dataloader
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import model_sharding
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import BColors, create_logger, format_numel_str, get_model_numel, requires_grad, to_torch_dtype
from opensora.utils.train_utils import MaskGenerator, create_colossalai_plugin, update_ema

SEARCH_BS_PREFIX = f"{BColors.OKGREEN}[Search BS]{BColors.ENDC}"


def main():
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs()
    assert cfg.dataset.type == "VariableVideoTextDataset", "Only VariableVideoTextDataset is supported"

    # == device and dtype ==
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

    # == colossalai init distributed training ==
    # NOTE: A very large timeout is set to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(cfg.get("seed", 1024))
    DistCoordinator()
    device = get_current_device()

    # == init logger, tensorboard & wandb ==
    logger = create_logger()

    # == init ColossalAI booster ==
    plugin = create_colossalai_plugin(
        plugin=cfg.get("plugin", "zero2"),
        dtype=cfg_dtype,
        grad_clip=cfg.get("grad_clip", 0),
        sp_size=cfg.get("sp_size", 1),
    )
    booster = Booster(plugin=plugin)

    # ======================================================
    # build model
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

    # == build diffusion model ==
    input_size = (None, None, None)
    latent_size = vae.get_latent_size(input_size)
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
        )
        .to(device, dtype)
        .train()
    )
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        "[Diffusion] Trainable model params: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel),
    )

    # == build ema for diffusion model ==
    ema = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema.eval()
    update_ema(ema, model, decay=0, sharded=False)

    # == setup loss function, build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # == setup optimizer ==
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        adamw_mode=True,
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("adam_eps", 1e-8),
    )
    lr_scheduler = None

    # == additional preparation ==
    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)
    if cfg.get("mask_ratios", None) is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)

    # =======================================================
    # distributed training preparation with colossalai
    # =======================================================
    logger.info("Preparing for distributed training...")
    # == boosting ==
    # NOTE: we set dtype first to make initialization of model consistent with the dtype; then reset it to the fp32 as we make diffusion scheduler in fp32
    torch.set_default_dtype(dtype)
    model, optimizer, _, _, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boosting model for distributed training")

    model_sharding(ema)

    def reset_optimizer():
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

    def build_dataset(resolution, num_frames, batch_size):
        bucket_config = {resolution: {num_frames: (1.0, batch_size)}}
        dataset = build_module(cfg.dataset, DATASETS)
        dataloader_args = dict(
            dataset=dataset,
            batch_size=None,
            num_workers=cfg.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            process_group=get_data_parallel_group(),
        )
        dataloader, sampler = prepare_dataloader(
            bucket_config=bucket_config,
            **dataloader_args,
        )
        num_batch = sampler.get_num_batch()
        num_steps_per_epoch = num_batch // dist.get_world_size()

        dataloader_iter = iter(dataloader)

        return dataloader_iter, num_steps_per_epoch, num_batch

    def train(resolution, num_frames, batch_size, warmup_steps=5, active_steps=5):
        logger.info(
            "%s Training resolution=%s, num_frames=%s, batch_size=%s",
            SEARCH_BS_PREFIX,
            resolution,
            num_frames,
            batch_size,
        )
        total_steps = warmup_steps + active_steps
        dataloader_iter, num_steps_per_epoch, num_batch = build_dataset(resolution, num_frames, batch_size)
        if num_batch == 0:  # no data
            logger.info("%s No data found for resolution=%s, num_frames=%s", SEARCH_BS_PREFIX, resolution, num_frames)
            return -1
        assert (
            num_steps_per_epoch >= total_steps
        ), f"num_steps_per_epoch={num_steps_per_epoch} < total_steps={total_steps}"
        duration = 0

        reset_optimizer()
        for step, batch in tqdm(
            enumerate(dataloader_iter),
            desc=f"({resolution},{num_frames}) bs={batch_size}",
            total=total_steps,
        ):
            if step >= total_steps:
                break
            if step >= warmup_steps:
                start = time.time()

            x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
            y = batch.pop("text")

            # == visual and text encoding ==
            with torch.no_grad():
                # Prepare visual inputs
                x = vae.encode(x)  # [B, C, T, H/P, W/P]
                # Prepare text inputs
                model_args = text_encoder.encode(y)

            # == mask ==
            mask = None
            if cfg.get("mask_ratios", None) is not None:
                mask = mask_generator.get_masks(x)
                model_args["x_mask"] = mask

            # == video meta info ==
            for k, v in batch.items():
                model_args[k] = v.to(device, dtype)

            # == diffusion loss computation ==
            loss_dict = scheduler.training_losses(model, x, model_args, mask=mask)

            # == backward & update ==
            loss = loss_dict["loss"].mean()
            booster.backward(loss=loss, optimizer=optimizer)
            optimizer.step()
            optimizer.zero_grad()

            # == update EMA ==
            update_ema(ema, model.module, optimizer=optimizer, decay=cfg.get("ema_decay", 0.9999))

            # == time accumulation ==
            if step >= warmup_steps:
                end = time.time()
                duration += end - start

        avg_step_time = duration / active_steps
        logger.info("%s Average step time: %.2f", SEARCH_BS_PREFIX, avg_step_time)
        return avg_step_time

    # =======================================================
    # search for bucket
    # =======================================================
    # == benchmark ==
    def benchmark(resolution, num_frames, lower_bound, upper_bound, ref_step_time=None):
        logger.info(
            "%s Benchmarking resolution=%s, num_frames=%s, lower_bound=%s, upper_bound=%s",
            SEARCH_BS_PREFIX,
            resolution,
            num_frames,
            lower_bound,
            upper_bound,
        )

        # binary search the largest valid batch size
        mid = target_batch_size = target_step_time = 0
        if ref_step_time is not None:
            min_dis = float("inf")
        while lower_bound <= upper_bound:
            mid = (lower_bound + upper_bound) // 2
            try:
                step_time = train(resolution, num_frames, mid)
                if step_time < 0:  # no data
                    return 0, 0
                if ref_step_time is not None:
                    if step_time < ref_step_time:
                        lower_bound = mid + 1
                        dis = abs(step_time - ref_step_time)
                        if dis < min_dis:
                            target_batch_size, target_step_time = mid, step_time
                            min_dis = dis
                    else:
                        upper_bound = mid - 1
                else:
                    target_batch_size, target_step_time = mid, step_time
                    lower_bound = mid + 1
            except Exception:
                traceback.print_exc()
                upper_bound = mid - 1

        logger.info(
            "%s Benchmarking result: batch_size=%s, step_time=%s", SEARCH_BS_PREFIX, target_batch_size, target_step_time
        )
        return target_batch_size, target_step_time

    # == build bucket ==
    bucket_config = cfg.bucket_config
    output_bucket_cfg = deepcopy(bucket_config)
    if cfg.get("resolution", None) is not None:
        bucket_config = {cfg.resolution: bucket_config[cfg.resolution]}
    buckets = {
        (resolution, num_frames): (max(guess_bs - variance, 1), guess_bs + variance)
        for resolution, t_bucket in bucket_config.items()
        for num_frames, (guess_bs, variance) in t_bucket.items()
    }

    # == get base_step_time ==
    base_step_time = cfg.get("base_step_time", None)
    result_table = []
    if base_step_time is None:
        base_resolution, base_num_frames = cfg.base
        base_num_frames = get_num_frames(base_num_frames)
        assert (
            base_resolution,
            base_num_frames,
        ) in buckets, f"Base bucket {base_resolution} {base_num_frames} not found"
        base_bound = buckets.pop((base_resolution, base_num_frames))

        base_batch_size, base_step_time = benchmark(base_resolution, base_num_frames, *base_bound)
        output_bucket_cfg[base_resolution][base_num_frames] = base_batch_size
        result_table.append(f"{base_resolution}, {base_num_frames}, {base_batch_size}, {base_step_time:.2f}")

    # == search for other buckets ==
    for (resolution, frames), bounds in buckets.items():
        if bounds[0] == bounds[1]:
            continue
        try:
            batch_size, step_time = benchmark(resolution, frames, *bounds, ref_step_time=base_step_time)
            output_bucket_cfg[resolution][frames] = batch_size
            result_table.append(f"{resolution}, {frames}, {batch_size}, {step_time:.2f}")
        except RuntimeError:
            pass
    result_table = "\n".join(result_table)
    logger.info("%s Search result:\nResolution, Frames, Batch size, Step time\n%s", SEARCH_BS_PREFIX, result_table)
    logger.info("%s Bucket searched: %s", SEARCH_BS_PREFIX, output_bucket_cfg)


if __name__ == "__main__":
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import argparse
import time

import torch
from colossalai import launch_from_torch
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from tqdm import tqdm

from open_sora.diffusion import create_diffusion
from open_sora.modeling import DiT_models
from open_sora.utils.data import create_video_compressor, preprocess_batch
from open_sora.utils.plugin import ZeroSeqParallelPlugin

#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    """
    Trains a new DiT model.
    """
    # init distributed environment
    launch_from_torch({})
    coordinator = DistCoordinator()
    logger = get_dist_logger()

    # set up acceleration plugins
    if args.plugin == "ddp":
        plugin = TorchDDPPlugin()
    elif args.plugin == "zero2":
        # use bf16 to avoid skipping the first few iterations due to NaNs
        plugin = ZeroSeqParallelPlugin(sp_size=args.sp_size, stage=2, precision="bf16")
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")
    booster = Booster(plugin=plugin)

    # Create video compressor
    video_compressor = create_video_compressor(args.compressor)
    model_kwargs = {
        "in_channels": video_compressor.out_channels,
        "seq_parallel_group": plugin.sp_group,
    }

    # Create DiT and EMA
    model = DiT_models[args.model](**model_kwargs).to(get_current_device())
    patch_size = model.patch_size
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # configure gradient checkpointing
    if args.grad_checkpoint:
        model.enable_gradient_checkpointing()

    # create diffusion pipeline
    diffusion = create_diffusion(
        timestep_respacing=""
    )  # default: 1000 steps, linear noise schedule

    # setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = HybridAdam(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup dataloader
    videos = [
        torch.randn(args.num_frames, args.height, args.width, 3)
        for _ in range(args.batch_size)
    ]
    assert args.num_tokens % args.sp_size == 0
    input_ids = torch.randn(args.batch_size, args.num_tokens, args.text_embed_dim)
    text_mask = torch.ones(input_ids.shape[:2], dtype=torch.int)
    batch = {
        "videos": videos,
        "text_latent_states": input_ids,
        "text_padding_mask": text_mask,
    }
    batch = preprocess_batch(
        batch, patch_size, video_compressor, pad_to_multiple=args.sp_size
    )
    video_inputs = batch.pop("video_latent_states")
    mask = batch.pop("video_padding_mask")
    logger.info(
        f"Num patches: {video_inputs.shape[1]}, num_tokens: {batch['text_latent_states'].shape[1]}",
        ranks=[0],
    )

    # setup booster
    model, opt, *_ = booster.boost(model, opt)
    logger.info(
        f"Booster init max device memory: {get_accelerator().max_memory_allocated() / 1024 ** 2:.2f} MB",
        ranks=[0],
    )

    # Train
    total_samples = 0
    total_duration = 0.0
    for i in tqdm(
        range(args.warmup_steps + args.steps),
        desc="Steps",
        disable=not coordinator.is_master(),
    ):
        start = time.time()
        t = torch.randint(
            0,
            diffusion.num_timesteps,
            (video_inputs.shape[0],),
            device=video_inputs.device,
        )
        loss_dict = diffusion.training_losses(model, video_inputs, t, batch, mask=mask)
        loss = loss_dict["loss"].mean()
        booster.backward(loss, opt)
        opt.step()
        opt.zero_grad()
        get_accelerator().empty_cache()
        time_per_iter = time.time() - start
        if i >= args.warmup_steps:
            total_samples += args.batch_size * coordinator.world_size
            total_duration += time_per_iter

    throughput = total_samples / total_duration
    logger.info(
        f"Training complete, max device memory: {get_accelerator().max_memory_allocated() / 1024 ** 2:.2f} MB",
        ranks=[0],
    )
    logger.info(
        f"Throughput per device: {throughput:.2f} samples/s",
        ranks=[0],
    )


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/8"
    )
    parser.add_argument(
        "-p", "--plugin", type=str, default="zero2", choices=["ddp", "zero2"]
    )
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("-w", "--warmup_steps", type=int, default=2)
    parser.add_argument("-s", "--steps", type=int, default=3)
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-f", "--num_frames", type=int, default=300)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--text_embed_dim", type=int, default=512)
    parser.add_argument("-g", "--grad_checkpoint", action="store_true", default=False)
    parser.add_argument(
        "-c", "--compressor", choices=["raw", "vqvae", "vae"], default="raw"
    )
    args = parser.parse_args()
    main(args)

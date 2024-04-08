import os

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype
from opensora.datasets import DatasetFromCSV, get_transforms_image, get_transforms_video, prepare_dataloader
from opensora.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)
from tqdm import tqdm
from opensora.models.vae.model_utils import VEA3DLoss


def main():
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False)
    print(cfg)

    # init distributed
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()

    # if coordinator.world_size > 1:
    #     set_sequence_parallel_group(dist.group.WORLD)
    #     enable_sequence_parallelism = True
    # else:
    #     enable_sequence_parallelism = False

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)


    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    dataset = DatasetFromCSV(
        cfg.data_path,
        # TODO: change transforms
        transform=(
            get_transforms_video(cfg.image_size[0])
            if not cfg.use_image_transform
            else get_transforms_image(cfg.image_size[0])
        ),
        num_frames=cfg.num_frames,
        frame_interval=cfg.frame_interval,
        root=cfg.root,
    )

    dataloader = prepare_dataloader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )
    print(f"Dataset contains {len(dataset):,} videos ({cfg.data_path})")

    total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.sp_size
    print(f"Total batch size: {total_batch_size}")

    # ======================================================
    # 4. build model & load weights
    # ======================================================
    # 3.1. build model
    # input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.model, MODELS)
    # latent_size = vae.get_latent_size(input_size)

    # 3.2. move to device & eval
    vae = vae.to(device, dtype).eval()

    # # 3.3. build scheduler
    # scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # 3.4. support for multi-resolution
    model_args = dict()
    if cfg.multi_resolution:
        image_size = cfg.image_size
        hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        model_args["data_info"] = dict(ar=ar, hw=hw)

    # ======================================================
    # 4. inference
    # ======================================================
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)


    # 4.1. batch generation
    
    # define loss function
    loss_function = VEA3DLoss(kl_weight=cfg.kl_weight, perceptual_weight=cfg.perceptual_weight).to(device, dtype)
    running_loss = 0.0
    loss_steps = 0


    total_steps = len(dataloader)
    dataloader_iter = iter(dataloader)

    with tqdm(
            range(total_steps),
            # desc=f"Avg Loss: {running_loss}",
            disable=not coordinator.is_master(),
            total=total_steps,
            initial=0,
        ) as pbar:
            for step in pbar:
                batch = next(dataloader_iter)
                x = batch["video"].to(device, dtype)  # [B, C, T, H, W]
                reconstructions, posterior = vae(x)
                loss = loss_function(x, reconstructions, posterior)
                loss_steps += 1
                running_loss = loss.item()/ loss_steps + running_loss * ((loss_steps - 1) / loss_steps)

            if coordinator.is_master():
                for idx, sample in enumerate(reconstructions):
                    save_path = os.path.join(save_dir, f"sample_{idx}")
                    save_sample(sample, fps=cfg.fps, save_path=save_path)

    print("test loss:", running_loss)
    

if __name__ == "__main__":
    main()

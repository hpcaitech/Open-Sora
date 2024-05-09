import os

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import IMG_FPS, save_sample
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype


def main():
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False)
    print(cfg)

    # init distributed
    if os.environ.get("WORLD_SIZE", None):
        use_dist = True
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()

        if coordinator.world_size > 1:
            set_sequence_parallel_group(dist.group.WORLD)
            enable_sequence_parallelism = True
        else:
            enable_sequence_parallelism = False
    else:
        use_dist = False
        enable_sequence_parallelism = False

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)
    prompts = cfg.prompt

    # ======================================================
    # 3. build model & load weights
    # ======================================================
    # 3.1. build model
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size(input_size)
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)  # T5 must be fp32

    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_sequence_parallelism=enable_sequence_parallelism,
    )
    text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    # 3.2. move to device & eval
    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()

    # 3.3. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # 3.4. support for multi-resolution
    model_args = dict()
    if cfg.multi_resolution == "PixArtMS":
        image_size = cfg.image_size
        hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        model_args["data_info"] = dict(ar=ar, hw=hw)
    elif cfg.multi_resolution == "STDiT2":
        image_size = cfg.image_size
        height = torch.tensor([image_size[0]], device=device, dtype=dtype).repeat(cfg.batch_size)
        width = torch.tensor([image_size[1]], device=device, dtype=dtype).repeat(cfg.batch_size)
        num_frames = torch.tensor([cfg.num_frames], device=device, dtype=dtype).repeat(cfg.batch_size)
        ar = torch.tensor([image_size[0] / image_size[1]], device=device, dtype=dtype).repeat(cfg.batch_size)
        if cfg.num_frames == 1:
            cfg.fps = IMG_FPS
        fps = torch.tensor([cfg.fps], device=device, dtype=dtype).repeat(cfg.batch_size)
        model_args["height"] = height
        model_args["width"] = width
        model_args["num_frames"] = num_frames
        model_args["ar"] = ar
        model_args["fps"] = fps

    # ======================================================
    # 4. inference
    # ======================================================
    sample_idx = 0
    if cfg.sample_name is not None:
        sample_name = cfg.sample_name
    elif cfg.prompt_as_path:
        sample_name = ""
    else:
        sample_name = "sample"
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 4.1. batch generation
    for i in range(0, len(prompts), cfg.batch_size):
        # 4.2 sample in hidden space
        batch_prompts_raw = prompts[i : i + cfg.batch_size]
        batch_prompts = [text_preprocessing(prompt) for prompt in batch_prompts_raw]
        # handle the last batch
        if len(batch_prompts_raw) < cfg.batch_size and cfg.multi_resolution == "STDiT2":
            model_args["height"] = model_args["height"][: len(batch_prompts_raw)]
            model_args["width"] = model_args["width"][: len(batch_prompts_raw)]
            model_args["num_frames"] = model_args["num_frames"][: len(batch_prompts_raw)]
            model_args["ar"] = model_args["ar"][: len(batch_prompts_raw)]
            model_args["fps"] = model_args["fps"][: len(batch_prompts_raw)]

        # 4.3. diffusion sampling
        old_sample_idx = sample_idx
        # generate multiple samples for each prompt
        for k in range(cfg.num_sample):
            sample_idx = old_sample_idx

            # Skip if the sample already exists
            # This is useful for resuming sampling VBench
            if cfg.prompt_as_path:
                skip = True
                for batch_prompt in batch_prompts_raw:
                    path = os.path.join(save_dir, f"{sample_name}{batch_prompt}")
                    if cfg.num_sample != 1:
                        path = f"{path}-{k}"
                    path = f"{path}.mp4"
                    if not os.path.exists(path):
                        skip = False
                        break
                if skip:
                    continue

            # sampling
            z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
            samples = scheduler.sample(
                model,
                text_encoder,
                z=z,
                prompts=batch_prompts,
                device=device,
                additional_args=model_args,
            )
            samples = vae.decode(samples.to(dtype))

            # 4.4. save samples
            if not use_dist or coordinator.is_master():
                for idx, sample in enumerate(samples):
                    print(f"Prompt: {batch_prompts_raw[idx]}")
                    if cfg.prompt_as_path:
                        sample_name_suffix = batch_prompts_raw[idx]
                    else:
                        sample_name_suffix = f"_{sample_idx}"
                    save_path = os.path.join(save_dir, f"{sample_name}{sample_name_suffix}")
                    if cfg.num_sample != 1:
                        save_path = f"{save_path}-{k}"
                    save_sample(sample, fps=cfg.fps // cfg.frame_interval, save_path=save_path)
                    sample_idx += 1


if __name__ == "__main__":
    main()

import os

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.utils import read_from_path
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype


def collect_references_batch(reference_paths, vae, image_size):
    refs_x = []
    for reference_path in reference_paths:
        ref_path = reference_path.split(";")
        ref = []
        for r_path in ref_path:
            r = read_from_path(r_path, image_size)
            r_x = vae.encode(r.unsqueeze(0).to(vae.device, vae.dtype))
            r_x = r_x.squeeze(0)
            ref.append(r_x)
        refs_x.append(ref)
    # refs_x: [batch, ref_num, C, T, H, W]
    return refs_x


def apply_mask_strategy(z, refs_x, mask_strategys, loop_i):
    masks = []
    for i, mask_strategy in enumerate(mask_strategys):
        mask_strategy = mask_strategy.split(";")
        mask = torch.ones(z.shape[2], dtype=torch.bool, device=z.device)
        for mst in mask_strategy:
            loop_id, m_id, m_ref_start, m_length, m_target_start = mst.split(",")
            loop_id = int(loop_id)
            if loop_id != loop_i:
                continue
            m_id = int(m_id)
            m_ref_start = int(m_ref_start)
            m_length = int(m_length)
            m_target_start = int(m_target_start)
            ref = refs_x[i][m_id]  # [C, T, H, W]
            if m_ref_start < 0:
                m_ref_start = ref.shape[1] + m_ref_start
            if m_target_start < 0:
                # z: [B, C, T, H, W]
                m_target_start = z.shape[2] + m_target_start
            z[i, :, m_target_start : m_target_start + m_length] = ref[:, m_ref_start : m_ref_start + m_length]
            mask[m_target_start : m_target_start + m_length] = 0
        masks.append(mask)
    masks = torch.stack(masks)
    return masks


def process_prompts(prompts, num_loop):
    ret_prompts = []
    for prompt in prompts:
        if prompt.startswith("|0|"):
            prompt_list = prompt.split("|")[1:]
            text_list = []
            for i in range(0, len(prompt_list), 2):
                start_loop = int(prompt_list[i])
                text = prompt_list[i + 1]
                end_loop = int(prompt_list[i + 2]) if i + 2 < len(prompt_list) else num_loop
                text_list.extend([text] * (end_loop - start_loop))
            assert len(text_list) == num_loop
            ret_prompts.append(text_list)
        else:
            ret_prompts.append([prompt] * num_loop)
    return ret_prompts


def main():
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False)
    print(cfg)

    # init distributed
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()

    if coordinator.world_size > 1:
        set_sequence_parallel_group(dist.group.WORLD)
        enable_sequence_parallelism = True
    else:
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
        dtype=dtype,
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
    if cfg.multi_resolution:
        image_size = cfg.image_size
        hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        model_args["data_info"] = dict(ar=ar, hw=hw)

    # 3.5 reference
    if cfg.reference_path is not None:
        assert len(cfg.reference_path) == len(prompts)
        assert len(cfg.reference_path) == len(cfg.mask_strategy)

    # ======================================================
    # 4. inference
    # ======================================================
    sample_idx = 0
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 4.1. batch generation
    for i in range(0, len(prompts), cfg.batch_size):
        batch_prompts_loops = process_prompts(prompts[i : i + cfg.batch_size], cfg.loop)
        video_clips = []

        # 4.2. load reference videos & images
        if cfg.reference_path is not None:
            refs_x = collect_references_batch(cfg.reference_path[i : i + cfg.batch_size], vae, cfg.image_size[0])
            mask_strategy = cfg.mask_strategy[i : i + cfg.batch_size]

        # 4.3. long video generation
        for loop_i in range(cfg.loop):
            # 4.4 sample in hidden space
            batch_prompts = [prompt[loop_i] for prompt in batch_prompts_loops]
            z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)

            # 4.5. apply mask strategy
            masks = None
            if cfg.reference_path is not None:
                if loop_i > 0:
                    ref_x = vae.encode(video_clips[-1])
                    for j, refs in enumerate(refs_x):
                        refs.append(ref_x[j])
                        mask_strategy[
                            j
                        ] += f";{loop_i},{len(refs)-1},-{cfg.condition_frame_length},{cfg.condition_frame_length},0"
                masks = apply_mask_strategy(z, refs_x, mask_strategy, loop_i)

            # 4.6. diffusion sampling
            samples = scheduler.sample(
                model,
                text_encoder,
                z=z,
                prompts=batch_prompts,
                device=device,
                additional_args=model_args,
                mask=masks,  # scheduler must support mask
            )
            samples = vae.decode(samples.to(dtype))
            video_clips.append(samples)

            # 4.7. save video
            if loop_i == cfg.loop - 1:
                if coordinator.is_master():
                    for idx in range(len(video_clips[0])):
                        video_clips_i = [video_clips[0][idx]] + [
                            video_clips[i][idx][:, cfg.condition_frame_length :] for i in range(1, cfg.loop)
                        ]
                        video = torch.cat(video_clips_i, dim=1)
                        print(f"Prompt: {prompts[i + idx]}")
                        save_path = os.path.join(save_dir, f"sample_{sample_idx}")
                        save_sample(video, fps=cfg.fps, save_path=save_path)
                        sample_idx += 1


if __name__ == "__main__":
    main()

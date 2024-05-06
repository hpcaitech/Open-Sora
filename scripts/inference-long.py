import json
import os
import re

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import IMG_FPS, save_sample
from opensora.datasets.utils import read_from_path
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype


def collect_references_batch(reference_paths, vae, image_size):
    refs_x = []
    for reference_path in reference_paths:
        if reference_path is None:
            refs_x.append([])
            continue
        ref_path = reference_path.split(";")
        ref = []
        for r_path in ref_path:
            r = read_from_path(r_path, image_size, transform_name="resize_crop")
            r_x = vae.encode(r.unsqueeze(0).to(vae.device, vae.dtype))
            r_x = r_x.squeeze(0)
            ref.append(r_x)
        refs_x.append(ref)
    # refs_x: [batch, ref_num, C, T, H, W]
    return refs_x


def process_mask_strategy(mask_strategy):
    mask_batch = []
    mask_strategy = mask_strategy.split(";")
    for mask in mask_strategy:
        mask_group = mask.split(",")
        assert len(mask_group) >= 1 and len(mask_group) <= 6, f"Invalid mask strategy: {mask}"
        if len(mask_group) == 1:
            mask_group.extend(["0", "0", "0", "1", "0"])
        elif len(mask_group) == 2:
            mask_group.extend(["0", "0", "1", "0"])
        elif len(mask_group) == 3:
            mask_group.extend(["0", "1", "0"])
        elif len(mask_group) == 4:
            mask_group.extend(["1", "0"])
        elif len(mask_group) == 5:
            mask_group.append("0")
        mask_batch.append(mask_group)
    return mask_batch


def apply_mask_strategy(z, refs_x, mask_strategys, loop_i):
    masks = []
    for i, mask_strategy in enumerate(mask_strategys):
        mask = torch.ones(z.shape[2], dtype=torch.float, device=z.device)
        if mask_strategy is None:
            masks.append(mask)
            continue
        mask_strategy = process_mask_strategy(mask_strategy)
        for mst in mask_strategy:
            loop_id, m_id, m_ref_start, m_target_start, m_length, edit_ratio = mst
            loop_id = int(loop_id)
            if loop_id != loop_i:
                continue
            m_id = int(m_id)
            m_ref_start = int(m_ref_start)
            m_length = int(m_length)
            m_target_start = int(m_target_start)
            edit_ratio = float(edit_ratio)
            ref = refs_x[i][m_id]  # [C, T, H, W]
            if m_ref_start < 0:
                m_ref_start = ref.shape[1] + m_ref_start
            if m_target_start < 0:
                # z: [B, C, T, H, W]
                m_target_start = z.shape[2] + m_target_start
            z[i, :, m_target_start : m_target_start + m_length] = ref[:, m_ref_start : m_ref_start + m_length]
            mask[m_target_start : m_target_start + m_length] = edit_ratio
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
                text = text_preprocessing(text)
                end_loop = int(prompt_list[i + 2]) if i + 2 < len(prompt_list) else num_loop
                text_list.extend([text] * (end_loop - start_loop))
            assert len(text_list) == num_loop, f"Prompt loop mismatch: {len(text_list)} != {num_loop}"
            ret_prompts.append(text_list)
        else:
            prompt = text_preprocessing(prompt)
            ret_prompts.append([prompt] * num_loop)
    return ret_prompts


def extract_json_from_prompts(prompts):
    additional_infos = []
    ret_prompts = []
    for prompt in prompts:
        parts = re.split(r"(?=[{\[])", prompt)
        assert len(parts) <= 2, f"Invalid prompt: {prompt}"
        ret_prompts.append(parts[0])
        if len(parts) == 1:
            additional_infos.append({})
        else:
            additional_infos.append(json.loads(parts[1]))
    return ret_prompts, additional_infos


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

    # 3.5 reference
    if cfg.reference_path is not None:
        assert len(cfg.reference_path) == len(
            prompts
        ), f"Reference path mismatch: {len(cfg.reference_path)} != {len(prompts)}"
        assert len(cfg.reference_path) == len(
            cfg.mask_strategy
        ), f"Mask strategy mismatch: {len(cfg.mask_strategy)} != {len(prompts)}"
    else:
        cfg.reference_path = [None] * len(prompts)
        cfg.mask_strategy = [None] * len(prompts)

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
        batch_prompts_raw = prompts[i : i + cfg.batch_size]
        batch_prompts_raw, additional_infos = extract_json_from_prompts(batch_prompts_raw)
        batch_prompts_loops = process_prompts(batch_prompts_raw, cfg.loop)
        # handle the last batch
        if len(batch_prompts_raw) < cfg.batch_size and cfg.multi_resolution == "STDiT2":
            model_args["height"] = model_args["height"][: len(batch_prompts_raw)]
            model_args["width"] = model_args["width"][: len(batch_prompts_raw)]
            model_args["num_frames"] = model_args["num_frames"][: len(batch_prompts_raw)]
            model_args["ar"] = model_args["ar"][: len(batch_prompts_raw)]
            model_args["fps"] = model_args["fps"][: len(batch_prompts_raw)]

        # 4.2. load reference videos & images
        for j, info in enumerate(additional_infos):
            if "reference_path" in info:
                cfg.reference_path[i + j] = info["reference_path"]
            if "mask_strategy" in info:
                cfg.mask_strategy[i + j] = info["mask_strategy"]
        refs_x = collect_references_batch(cfg.reference_path[i : i + cfg.batch_size], vae, cfg.image_size)
        mask_strategy = cfg.mask_strategy[i : i + cfg.batch_size]

        # 4.3. diffusion sampling
        old_sample_idx = sample_idx
        # generate multiple samples for each prompt
        for k in range(cfg.num_sample):
            sample_idx = old_sample_idx
            video_clips = []

            # 4.4. long video generation
            for loop_i in range(cfg.loop):
                # 4.4 sample in hidden space
                batch_prompts = [prompt[loop_i] for prompt in batch_prompts_loops]

                # 4.5. apply mask strategy
                masks = None
                # if cfg.reference_path is not None:
                if loop_i > 0:
                    ref_x = vae.encode(video_clips[-1])
                    for j, refs in enumerate(refs_x):
                        if refs is None:
                            refs_x[j] = [ref_x[j]]
                        else:
                            refs.append(ref_x[j])
                        if mask_strategy[j] is None:
                            mask_strategy[j] = ""
                        else:
                            mask_strategy[j] += ";"
                        mask_strategy[
                            j
                        ] += f"{loop_i},{len(refs)-1},-{cfg.condition_frame_length},0,{cfg.condition_frame_length}"

                # sampling
                z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
                masks = apply_mask_strategy(z, refs_x, mask_strategy, loop_i)
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
                    if not use_dist or coordinator.is_master():
                        for idx in range(len(video_clips[0])):
                            video_clips_i = [video_clips[0][idx]] + [
                                video_clips[i][idx][:, cfg.condition_frame_length :] for i in range(1, cfg.loop)
                            ]
                            video = torch.cat(video_clips_i, dim=1)
                            print(f"Prompt: {batch_prompts_raw[idx]}")
                            if cfg.prompt_as_path:
                                sample_name_suffix = batch_prompts_raw[idx]
                            else:
                                sample_name_suffix = f"_{sample_idx}"
                            save_path = os.path.join(save_dir, f"{sample_name}{sample_name_suffix}")
                            if cfg.num_sample != 1:
                                save_path = f"{save_path}-{k}"
                            save_sample(video, fps=cfg.fps // cfg.frame_interval, save_path=save_path)
                            sample_idx += 1


if __name__ == "__main__":
    main()

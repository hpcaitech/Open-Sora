import os
import time
from pprint import pformat

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import (
    add_watermark,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    deflicker,
    extract_images_from_ref_paths,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prep_ref_and_mask,
    prep_ref_and_update_mask_in_loop,
    prepare_multi_resolution_info,
    print_memory_usage,
    refine_batched_prompts_with_images,
    refine_prompts_by_openai,
    split_prompt,
    super_resolution,
)
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype


def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    else:
        coordinator = None
        enable_sequence_parallelism = False
    set_random_seed(seed=cfg.get("seed", 1024))

    # == init logger ==
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

    # == prepare video size ==
    image_size = cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)

    # == build diffusion model ==
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
        .to(device, dtype)
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ======================================================
    # inference
    # ======================================================
    # == load prompts ==
    prompts = cfg.get("prompt", None)
    start_idx = cfg.get("start_index", 0)
    obtained_ref_from_csv = False
    if prompts is None:
        if cfg.get("prompt_path", None) is not None:
            if cfg.get("csv_reference_column_name") is not None and cfg.prompt_path.endswith("csv"):
                prompts, reference_path = load_prompts(
                    cfg.prompt_path,
                    start_idx,
                    cfg.get("end_index", None),
                    csv_ref_column_name=cfg.csv_reference_column_name,
                )
                obtained_ref_from_csv = True
            else:
                prompts = load_prompts(cfg.prompt_path, start_idx, cfg.get("end_index", None))
        else:
            prompts = [cfg.get("prompt_generator", "")] * 1_000_000  # endless loop

    # == prepare reference ==
    neg_prompts = cfg.get("neg_prompt", [None] * len(prompts))

    if not obtained_ref_from_csv:
        reference_path = cfg.get("reference_path", [""] * len(prompts))

    mask_strategy = cfg.get("mask_strategy", [""] * len(prompts))
    assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
    assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

    # == prepare arguments ==
    fps = cfg.fps
    save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
    multi_resolution = cfg.get("multi_resolution", None)
    batch_size = cfg.get("batch_size", 1)
    num_sample = cfg.get("num_sample", 1)
    loop = cfg.get("loop", 1)
    condition_frame_length = cfg.get("condition_frame_length", 5)
    cfg.get("condition_frame_edit", 0.0)
    align = cfg.get("align", None)

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    sample_name = cfg.get("sample_name", None)
    prompt_as_path = cfg.get("prompt_as_path", False)

    use_sdedit = cfg.get("use_sdedit", False)
    use_oscillation_guidance_for_text = cfg.get("use_oscillation_guidance_for_text", None)
    use_oscillation_guidance_for_image = cfg.get("use_oscillation_guidance_for_image", None)

    # == Iter over all samples ==
    for i in progress_wrap(range(0, len(prompts), batch_size)):
        # == prepare batch prompts ==
        batch_prompts = prompts[i : i + batch_size]
        ms = mask_strategy[i : i + batch_size]
        batch_reference_paths = reference_path[i : i + batch_size]
        neg_prompts_batch = neg_prompts[i : i + batch_size]

        # == get json from prompts ==
        if not obtained_ref_from_csv:
            batch_prompts, batch_reference_paths, ms = extract_json_from_prompts(
                batch_prompts, batch_reference_paths, ms
            )

        original_batch_prompts = batch_prompts

        # == get reference for condition ==
        refs = collect_references_batch(batch_reference_paths, vae, image_size)

        # == multi-resolution info ==
        model_args = prepare_multi_resolution_info(
            multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
        )

        # == Iter over number of sampling for one prompt ==
        for k in range(num_sample):
            # == prepare save paths ==
            save_paths = [
                get_save_path_name(
                    save_dir,
                    sample_name=sample_name,
                    sample_idx=start_idx + idx,
                    prompt=original_batch_prompts[idx],
                    prompt_as_path=prompt_as_path,
                    num_sample=num_sample,
                    k=k,
                )
                for idx in range(len(batch_prompts))
            ]

            # NOTE: Skip if the sample already exists
            # This is useful for resuming sampling VBench
            if prompt_as_path and all_exists(save_paths):
                continue

            # == process prompts step by step ==
            # 0. split prompt
            # each element in the list is [prompt_segment_list, loop_idx_list]
            batched_prompt_segment_list = []
            batched_loop_idx_list = []
            for prompt in batch_prompts:
                prompt_segment_list, loop_idx_list = split_prompt(prompt)
                batched_prompt_segment_list.append(prompt_segment_list)
                batched_loop_idx_list.append(loop_idx_list)

            # 1. refine prompt by openai
            if cfg.get("llm_refine", False):
                # only call openai API when
                # 1. seq parallel is not enabled
                # 2. seq parallel is enabled and the process is rank 0
                if not enable_sequence_parallelism or (enable_sequence_parallelism and is_main_process()):
                    for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                        # batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)
                        images = extract_images_from_ref_paths(batch_reference_paths, image_size)
                        images = [img[0] for img in images]
                        batched_prompt_segment_list[idx] = refine_batched_prompts_with_images(
                            prompt_segment_list, images
                        )
                        batched_prompt_segment_list[idx] = refine_prompts_by_openai(batched_prompt_segment_list[idx])

                # sync the prompt if using seq parallel
                if enable_sequence_parallelism:
                    coordinator.block_all()
                    prompt_segment_length = [
                        len(prompt_segment_list) for prompt_segment_list in batched_prompt_segment_list
                    ]

                    # flatten the prompt segment list
                    batched_prompt_segment_list = [
                        prompt_segment
                        for prompt_segment_list in batched_prompt_segment_list
                        for prompt_segment in prompt_segment_list
                    ]

                    # create a list of size equal to world size
                    broadcast_obj_list = [batched_prompt_segment_list] * coordinator.world_size
                    dist.broadcast_object_list(broadcast_obj_list, 0)

                    # recover the prompt list
                    batched_prompt_segment_list = []
                    segment_start_idx = 0
                    all_prompts = broadcast_obj_list[0]
                    for num_segment in prompt_segment_length:
                        batched_prompt_segment_list.append(
                            all_prompts[segment_start_idx : segment_start_idx + num_segment]
                        )
                        segment_start_idx += num_segment

            # 2. append score
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = append_score_to_prompts(
                    prompt_segment_list,
                    aes=cfg.get("aes", None),
                    flow=cfg.get("flow", None),
                    camera_motion=cfg.get("camera_motion", None),
                )

            # 3. clean prompt with T5
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]
            if neg_prompts_batch[0] is None:
                neg_prompts_batch_cl = None
            else:
                neg_prompts_batch_cl = [text_preprocessing(prompt) for prompt in neg_prompts_batch]

            # 4. merge to obtain the final prompt
            batch_prompts = []
            for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

            # == Iter over loop generation ==
            ref = None
            cond_type = cfg.get("cond_type", None)
            image_cfg_scale = None
            image_cfg_scale = cfg.get("image_cfg_scale", 7.5)
            target_shape = [len(batch_prompts), vae.out_channels, *latent_size]
            ref_len = [len(ref) for ref in refs]
            if 0 in ref_len:
                print(f"refenrece not provided for {cond_type}, will default to t2v generation")
                cond_type = None

            ref, mask_index = prep_ref_and_mask(
                cond_type, condition_frame_length, refs, target_shape, loop, device, dtype
            )

            video_clips = []
            for loop_i in range(loop):
                # == get prompt for loop i ==
                batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)
                # == sampling ==
                z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
                masks = (
                    apply_mask_strategy(z, refs, ms, loop_i, align=align) if len(mask_index) == 0 else None
                )  # no mask for i2v and v2v
                x_cond_mask = torch.zeros(len(batch_prompts), vae.out_channels, *latent_size, device=device).to(dtype)
                if len(mask_index) > 0:
                    x_cond_mask[:, :, mask_index, :, :] = 1.0

                samples = scheduler.sample(
                    model,
                    text_encoder,
                    z=z,
                    z_cond=ref,
                    z_cond_mask=x_cond_mask,
                    prompts=batch_prompts_loop,
                    device=device,
                    additional_args=model_args,
                    progress=verbose >= 2,
                    mask=masks,
                    mask_index=mask_index,
                    image_cfg_scale=image_cfg_scale,
                    neg_prompts=neg_prompts_batch_cl if len(mask_index) > 0 else None,  # no mask for i2v and v2v
                    use_sdedit=use_sdedit,
                    use_oscillation_guidance_for_text=use_oscillation_guidance_for_text,
                    use_oscillation_guidance_for_image=use_oscillation_guidance_for_image,
                )

                is_last_loop = loop_i == loop - 1
                if loop > 1 and not is_last_loop:  # process conditions for subsequent loop
                    ref, mask_index = prep_ref_and_update_mask_in_loop(
                        cond_type,
                        condition_frame_length,
                        samples,
                        refs,
                        target_shape,
                        is_last_loop,
                        device,
                        dtype,
                    )

                video_clips.append(samples)

            # == save samples ==
            if is_main_process():
                for idx, batch_prompt in enumerate(batch_prompts):
                    if verbose >= 2:
                        logger.info("Prompt: %s", batch_prompt)
                    save_path = save_paths[idx]
                    video = [video_clips[i][idx] for i in range(loop)]
                    for i in range(1, loop):
                        video[i] = video[i][:, condition_frame_length:]  # latent video concat
                    video = torch.cat(video, dim=1)  # latent [C, T, H, W]
                    # ensure latent frame size is multiples of 5
                    t_cut = video.size(1) // 5 * 5
                    if t_cut < video.size(1):
                        video = video[:, :t_cut]

                    video = vae.decode(video.to(dtype), num_frames=t_cut * 17 // 5).squeeze(0)

                    print("video size:", video.size())
                    save_path = save_sample(
                        video,
                        fps=save_fps,
                        save_path=save_path,
                        verbose=verbose >= 2,
                    )
                    if save_path.endswith(".mp4") and cfg.get("deflicker", False):
                        time.sleep(1)
                        save_path = deflicker(save_path)
                    if save_path.endswith(".mp4") and cfg.get("super_resolution", False):
                        time.sleep(1)
                        save_path = super_resolution(save_path, cfg.get("super_resolution"))
                    if save_path.endswith(".mp4") and cfg.get("watermark", False):
                        time.sleep(1)  # prevent loading previous generated video
                        add_watermark(save_path)
        start_idx += len(batch_prompts)
    logger.info("Inference finished.")
    logger.info("Saved %s samples to %s", start_idx, save_dir)
    print_memory_usage("After inference", device)


if __name__ == "__main__":
    main()

import os
import time
from datetime import date, datetime, timedelta
from pprint import pformat

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_num_frames
from opensora.registry import MODELS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import add_watermark, dframe_to_frame, load_prompts
from opensora.utils.misc import create_logger, is_distributed, is_main_process, to_torch_dtype


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
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

    # ======================================================
    # inference
    # ======================================================
    # == load prompts ==
    prompts = cfg.get("prompt", None)
    start_idx = cfg.get("start_index", 0)
    if prompts is None:
        if cfg.get("prompt_path", None) is not None:
            prompts = load_prompts(cfg.prompt_path, start_idx, cfg.get("end_index", None))
        else:
            prompts = [cfg.get("prompt_generator", "")] * 1_000_000  # endless loop

    # == prepare reference ==
    reference_path = cfg.get("reference_path", [""] * len(prompts))
    mask_strategy = cfg.get("mask_strategy", [""] * len(prompts))
    assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
    assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

    # == prepare arguments ==
    num_frames = get_num_frames(cfg.num_frames)
    fps = cfg.fps
    save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
    cfg.get("multi_resolution", None)
    batch_size = cfg.get("batch_size", 1)
    cfg.get("num_sample", 1)
    loop = cfg.get("loop", 1)
    condition_frame_length = cfg.get("condition_frame_length", 5)
    cfg.get("condition_frame_edit", 0.0)
    cfg.get("align", None)

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    cfg.get("sample_name", None)
    cfg.get("prompt_as_path", False)

    # == prepare saved dir ==
    cur_date = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(os.path.join(save_dir, cur_date)):
        yesterday = date.today() - timedelta(days=1)
        cur_date = yesterday.strftime("%Y-%m-%d")
    latest_idx = sorted([int(x) for x in os.listdir(os.path.join(save_dir, cur_date))])[-1]
    saved_idx = str(latest_idx).zfill(5)

    # == Iter over all samples ==
    for i in progress_wrap(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i : i + batch_size]
        # == Iter over loop generation ==
        video_clips = []
        for loop_i in range(loop):
            # == get prompt for loop i ==
            samples = torch.load(os.path.join(save_dir, cur_date, saved_idx, f"{i}_{loop_i}_latents.pt"))
            samples = samples.to(device, dtype)
            samples = vae.decode(samples.to(dtype), num_frames=num_frames)
            video_clips.append(samples)

        # == save samples ==
        if is_main_process():
            for idx, batch_prompt in enumerate(batch_prompts):
                if verbose >= 2:
                    logger.info("Prompt: %s", batch_prompt)
                save_path = os.path.join(save_dir, cur_date, saved_idx, "video")
                video = [video_clips[i][idx] for i in range(loop)]
                for i in range(1, loop):
                    video[i] = video[i][:, dframe_to_frame(condition_frame_length) :]
                video = torch.cat(video, dim=1)
                save_path = save_sample(
                    video,
                    fps=save_fps,
                    save_path=save_path,
                    verbose=verbose >= 2,
                )
                if save_path.endswith(".mp4") and cfg.get("watermark", False):
                    time.sleep(1)  # prevent loading previous generated video
                    add_watermark(save_path)
        start_idx += len(batch_prompts)
    logger.info("Inference VAE decoder finished.")
    logger.info("Saved %s samples to %s/%s/%s", start_idx, save_dir, cur_date, saved_idx)


if __name__ == "__main__":
    main()

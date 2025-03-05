import os
import time
import warnings
from pprint import pformat

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.distributed as dist
from colossalai.utils import set_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.registry import DATASETS, build_module
from opensora.utils.cai import (
    get_booster,
    get_is_saving_process,
    init_inference_environment,
)
from opensora.utils.config import parse_alias, parse_configs
from opensora.utils.inference import (
    add_fps_info_to_text,
    add_motion_score_to_text,
    create_tmp_csv,
    modify_option_to_t2i,
    process_and_save,
)
from opensora.utils.logger import create_logger, is_main_process
from opensora.utils.misc import log_cuda_max_memory, to_torch_dtype
from opensora.utils.prompt_refine import refine_prompts
from opensora.utils.sampling import (
    SamplingOption,
    prepare_api,
    prepare_models,
    sanitize_sampling_option,
)


@torch.inference_mode()
def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    torch.set_grad_enabled(False)

    # == parse configs ==
    cfg = parse_configs()
    cfg = parse_alias(cfg)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    seed = cfg.get("seed", 1024)
    if seed is not None:
        set_seed(seed)

    # == init distributed env ==
    init_inference_environment()
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    is_saving_process = get_is_saving_process(cfg)
    booster = get_booster(cfg)
    booster_ae = get_booster(cfg, ae=True)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")

    # save directory
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # == build dataset ==
    if cfg.get("prompt"):
        cfg.dataset.data_path = create_tmp_csv(save_dir, cfg.prompt, cfg.get("ref", None), create=is_main_process())
    dist.barrier()
    dataset = build_module(cfg.dataset, DATASETS)

    # range selection
    start_index = cfg.get("start_index", 0)
    end_index = cfg.get("end_index", None)
    if end_index is None:
        end_index = start_index + cfg.get("num_samples", len(dataset.data) + 1)
    dataset.data = dataset.data[start_index:end_index]
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", 1),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
    )
    dataloader, _ = prepare_dataloader(**dataloader_args)

    # == prepare default params ==
    sampling_option = SamplingOption(**cfg.sampling_option)
    sampling_option = sanitize_sampling_option(sampling_option)

    cond_type = cfg.get("cond_type", "t2v")
    prompt_refine = cfg.get("prompt_refine", False)
    fps_save = cfg.get("fps_save", 16)
    num_sample = cfg.get("num_sample", 1)

    type_name = "image" if cfg.sampling_option.num_frames == 1 else "video"
    sub_dir = f"{type_name}_{cfg.sampling_option.resolution}"
    os.makedirs(os.path.join(save_dir, sub_dir), exist_ok=True)
    use_t2i2v = cfg.get("use_t2i2v", False)
    img_sub_dir = os.path.join(sub_dir, "generated_condition")
    if use_t2i2v:
        os.makedirs(os.path.join(save_dir, sub_dir, "generated_condition"), exist_ok=True)

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")

    # == build flux model ==
    model, model_ae, model_t5, model_clip, optional_models = prepare_models(
        cfg, device, dtype, offload_model=cfg.get("offload_model", False)
    )
    log_cuda_max_memory("build model")

    if booster:
        model, _, _, _, _ = booster.boost(model=model)
        model = model.unwrap()
    if booster_ae:
        model_ae, _, _, _, _ = booster_ae.boost(model=model_ae)
        model_ae = model_ae.unwrap()

    api_fn = prepare_api(model, model_ae, model_t5, model_clip, optional_models)

    # prepare image flux model if t2i2v
    if use_t2i2v:
        api_fn_img = prepare_api(
            optional_models["img_flux"], optional_models["img_flux_ae"], model_t5, model_clip, optional_models
        )

    # ======================================================
    # 4. inference
    # ======================================================
    for epoch in range(num_sample):  # generate multiple samples with different seeds
        dataloader_iter = iter(dataloader)
        with tqdm(
            enumerate(dataloader_iter, start=0),
            desc="Inference progress",
            disable=not is_main_process(),
            initial=0,
            total=len(dataloader),
        ) as pbar:
            for _, batch in pbar:
                original_text = batch.pop("text")
                if use_t2i2v:
                    batch["text"] = original_text if not prompt_refine else refine_prompts(original_text, type="t2i")
                    sampling_option_t2i = modify_option_to_t2i(
                        sampling_option,
                        distilled=True,
                        img_resolution=cfg.get("img_resolution", "768px"),
                    )
                    if cfg.get("offload_model", False):
                        model_move_start = time.time()
                        model = model.to("cpu", dtype)
                        model_ae = model_ae.to("cpu", dtype)
                        optional_models["img_flux"].to(device, dtype)
                        optional_models["img_flux_ae"].to(device, dtype)
                        logger.info(
                            "offload video diffusion model to cpu, load image flux model to gpu: %s s",
                            time.time() - model_move_start,
                        )

                    logger.info("Generating image condition by flux...")
                    x_cond = api_fn_img(
                        sampling_option_t2i,
                        "t2v",
                        seed=sampling_option.seed + epoch if sampling_option.seed else None,
                        channel=cfg["img_flux"]["in_channels"],
                        **batch,
                    ).cpu()

                    # save image to disk
                    batch["name"] = process_and_save(
                        x_cond,
                        batch,
                        cfg,
                        img_sub_dir,
                        sampling_option_t2i,
                        epoch,
                        start_index,
                        saving=is_saving_process,
                    )
                    dist.barrier()

                    if cfg.get("offload_model", False):
                        model_move_start = time.time()
                        model = model.to(device, dtype)
                        model_ae = model_ae.to(device, dtype)
                        optional_models["img_flux"].to("cpu", dtype)
                        optional_models["img_flux_ae"].to("cpu", dtype)
                        logger.info(
                            "load video diffusion model to gpu, offload image flux model to cpu: %s s",
                            time.time() - model_move_start,
                        )

                    ref_dir = os.path.join(save_dir, os.path.join(sub_dir, "generated_condition"))
                    batch["ref"] = [os.path.join(ref_dir, f"{x}.png") for x in batch["name"]]
                    cond_type = "i2v_head"

                batch["text"] = original_text
                if prompt_refine:
                    batch["text"] = refine_prompts(
                        original_text, type="t2v" if cond_type == "t2v" else "t2i", image_paths=batch.get("ref", None)
                    )
                batch["text"] = add_fps_info_to_text(batch.pop("text"), fps=fps_save)
                if "motion_score" in cfg:
                    batch["text"] = add_motion_score_to_text(batch.pop("text"), cfg.get("motion_score", 5))

                logger.info("Generating video...")
                x = api_fn(
                    sampling_option,
                    cond_type,
                    seed=sampling_option.seed + epoch if sampling_option.seed else None,
                    patch_size=cfg.get("patch_size", 2),
                    save_prefix=cfg.get("save_prefix", ""),
                    channel=cfg["model"]["in_channels"],
                    **batch,
                ).cpu()

                if is_saving_process:
                    process_and_save(x, batch, cfg, sub_dir, sampling_option, epoch, start_index)
                dist.barrier()

    logger.info("Inference finished.")
    log_cuda_max_memory("inference")


if __name__ == "__main__":
    main()

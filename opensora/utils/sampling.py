import math
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
import json
import subprocess
from collections import defaultdict
import sys

import torch
import torchvision
from einops import rearrange, repeat
from mmengine.config import Config
from peft import PeftModel
from torch import Tensor, nn
import torch.distributed as dist

from opensora.datasets.aspect import get_image_size
from opensora.models.mmdit.model import MMDiTModel
from opensora.models.text.conditioner import HFEmbedder
from opensora.registry import MODELS, build_module
from opensora.utils.inference import (
    SamplingMethod,
    collect_references_batch,
    prepare_inference_condition,
)

# ======================================================
# Sampling Options
# ======================================================


@dataclass
class SamplingOption:
    # The width of the image/video.
    width: int | None = None

    # The height of the image/video.
    height: int | None = None

    # The resolution of the image/video. If provided, it will override the height and width.
    resolution: str | None = None

    # The aspect ratio of the image/video. If provided, it will override the height and width.
    aspect_ratio: str | None = None

    # The number of frames.
    num_frames: int = 1

    # The number of sampling steps.
    num_steps: int = 50

    # The classifier-free guidance (text).
    guidance: float = 4.0

    # use oscillation for text guidance
    text_osci: bool = False

    # The classifier-free guidance (image), or for the guidance on condition for i2v and v2v
    guidance_img: float | None = None

    # use oscillation for image guidance
    image_osci: bool = False

    # use temporal scaling for image guidance
    scale_temporal_osci: bool = False

    # The seed for the random number generator.
    seed: int | None = None

    # Whether to shift the schedule.
    shift: bool = True

    # The sampling method.
    method: str | SamplingMethod = SamplingMethod.I2V

    # Temporal reduction
    temporal_reduction: int = 1

    # is causal vae
    is_causal_vae: bool = False

    # flow shift
    flow_shift: float | None = None

    # do_inference_scaling
    do_inference_scaling: bool | None = False

    # num_subtree
    num_subtree: int = 3

    # backward_scale
    backward_scale: float = 1.0

    # forward_scale
    forward_scale: float = 1.0

    # scaling_steps
    scaling_steps: list = None

    # vbench_gpus
    vbench_gpus: list = None

    # vbench_dimension_list
    vbench_dimension_list: list = None


NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
}

DIM_WEIGHT = {
    "subject consistency":1,
    "background consistency":1,
    "motion smoothness":1,
    "aesthetic quality":1,
    "imaging quality":1,
    "dynamic degree":0.5,
}


def find_highest_score_video(data):
    video_scores = defaultdict(dict)
    
    for metric_name, metric_data in data.items():
        if not isinstance(metric_data, list) or len(metric_data) < 2:
            continue
            
        if metric_name not in NORMALIZE_DIC:
            continue
            
        min_val = NORMALIZE_DIC[metric_name]["Min"] 
        max_val = NORMALIZE_DIC[metric_name]["Max"]
        dim_weight = DIM_WEIGHT[metric_name]
            
        for entry in metric_data[1]:
            try:
                path_parts = entry["video_path"].split("/")
                filename = path_parts[-1]
                video_index = int(filename.split(".")[0])
                
                if "video_results" in entry:
                    raw_score = entry["video_results"]
                elif "cor_num_per_video" in entry:
                    raw_score = entry["cor_num_per_video"]
                else:
                    continue
                    
                norm_score = (raw_score - min_val) / (max_val - min_val) * dim_weight
                video_scores[video_index][metric_name] = norm_score
                
            except (KeyError, ValueError, IndexError):
                continue

    final_scores = {}
    for vid, scores in video_scores.items():
        if len(scores) > 0:
            final_scores[vid] = sum(scores.values()) / sum(DIM_WEIGHT[key] for key in scores.keys())
    
    if not final_scores:
        return -1
        
    max_score = max(final_scores.values())
    candidates = [vid for vid, score in final_scores.items() if score == max_score]
    return min(candidates) if candidates else -1


def sanitize_sampling_option(sampling_option: SamplingOption) -> SamplingOption:
    """
    Sanitize the sampling options.

    Args:
        sampling_option (SamplingOption): The sampling options.

    Returns:
        SamplingOption: The sanitized sampling options.
    """
    if (
        sampling_option.resolution is not None
        or sampling_option.aspect_ratio is not None
    ):
        assert (
            sampling_option.resolution is not None
            and sampling_option.aspect_ratio is not None
        ), "Both resolution and aspect ratio must be provided"
        resolution = sampling_option.resolution
        aspect_ratio = sampling_option.aspect_ratio
        height, width = get_image_size(resolution, aspect_ratio, training=False)
    else:
        assert (
            sampling_option.height is not None and sampling_option.width is not None
        ), "Both height and width must be provided"
        height, width = sampling_option.height, sampling_option.width

    height = (height // 16 + (1 if height % 16 else 0)) * 16
    width = (width // 16 + (1 if width % 16 else 0)) * 16
    replace_dict = dict(height=height, width=width)

    if isinstance(sampling_option.method, str):
        method = SamplingMethod(sampling_option.method)
        replace_dict["method"] = method

    return replace(sampling_option, **replace_dict)


def get_oscillation_gs(guidance_scale: float, i: int, force_num=10):
    """
    get oscillation guidance for cfg.

    Args:
        guidance_scale: original guidance value
        i: denoising step
        force_num: before which don't apply oscillation
    """
    if i < force_num or (i >= force_num and i % 2 == 0):
        gs = guidance_scale
    else:
        gs = 1.0
    return gs


# ======================================================
# Denoising
# ======================================================


class Denoiser(ABC):
    @abstractmethod
    def denoise(self, model: MMDiTModel, **kwargs) -> Tensor:
        """Denoise the input."""

    @abstractmethod
    def prepare_guidance(
        self,
        text: list[str],
        optional_models: dict[str, nn.Module],
        device: torch.device,
        dtype: torch.dtype,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Prepare the guidance for the model. This method will alter text."""


class I2VDenoiser(Denoiser):
    def denoise(self, model: MMDiTModel, **kwargs) -> Tensor:
        img = kwargs.pop("img")
        timesteps = kwargs.pop("timesteps")
        guidance = kwargs.pop("guidance")
        guidance_img = kwargs.pop("guidance_img")

        # cond ref arguments
        masks = kwargs.pop("masks")
        masked_ref = kwargs.pop("masked_ref")
        kwargs.pop("sigma_min")

        # oscillation guidance
        text_osci = kwargs.pop("text_osci", False)
        image_osci = kwargs.pop("image_osci", False)
        scale_temporal_osci = kwargs.pop("scale_temporal_osci", False)

        # patch size
        patch_size = kwargs.pop("patch_size", 2)

        guidance_vec = torch.full(
            (img.shape[0],), guidance, device=img.device, dtype=img.dtype
        )
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            # timesteps
            t_vec = torch.full(
                (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
            )
            b, c, t, w, h = masked_ref.size()
            cond = torch.cat((masks, masked_ref), dim=1)
            cond = pack(cond, patch_size=patch_size)
            kwargs["cond"] = torch.cat([cond, cond, torch.zeros_like(cond)], dim=0)

            # forward preparation
            cond_x = img[: len(img) // 3]

            img = torch.cat([cond_x, cond_x, cond_x], dim=0)
            # forward
            pred = model(
                img=img,
                **kwargs,
                timesteps=t_vec,
                guidance=guidance_vec,
            )

            # prepare guidance
            text_gs = get_oscillation_gs(guidance, i) if text_osci else guidance
            image_gs = (
                get_oscillation_gs(guidance_img, i) if image_osci else guidance_img
            )
            cond, uncond, uncond_2 = pred.chunk(3, dim=0)
            if image_gs > 1.0 and scale_temporal_osci:
                # image_gs decrease with each denoising step
                step_upper_image_gs = torch.linspace(image_gs, 1.0, len(timesteps))[i]
                # image_gs increase along the temporal axis of the latent video
                image_gs = torch.linspace(1.0, step_upper_image_gs, t)[
                    None, None, :, None, None
                ].repeat(b, c, 1, h, w)
                image_gs = pack(image_gs, patch_size=patch_size).to(cond.device, cond.dtype)

            # update
            pred = uncond_2 + image_gs * (uncond - uncond_2) + text_gs * (cond - uncond)
            pred = torch.cat([pred, pred, pred], dim=0)

            img = img + (t_prev - t_curr) * pred

        img = img[: len(img) // 3]

        return img

    def prepare_guidance(
        self,
        text: list[str],
        optional_models: dict[str, nn.Module],
        device: torch.device,
        dtype: torch.dtype,
        **kwargs,
    ) -> tuple[list[str], dict[str, Tensor]]:
        ret = {}

        neg = kwargs.get("neg", None)
        ret["guidance_img"] = kwargs.pop("guidance_img")

        # text
        if neg is None:
            neg = [""] * len(text)
        text = text + neg + neg
        return text, ret


class I2VScalingDenoiser(Denoiser):
    def denoise(self, model: MMDiTModel, **kwargs) -> Tensor:
        img = kwargs.pop("img")
        timesteps = kwargs.pop("timesteps")
        guidance = kwargs.pop("guidance")
        guidance_img = kwargs.pop("guidance_img")
        vbench = kwargs.pop("vbench", None)
        do_inference_scaling = kwargs.pop("do_inference_scaling", True)
        num_subtree = kwargs.pop("num_subtree", 3)
        model_ae = kwargs.pop("model_ae", None)
        height = kwargs.pop("height", None)
        width = kwargs.pop("width", None)
        num_frames = kwargs.pop("num_frames", None)
        patch_size = kwargs.pop("patch_size", None)
        prompt = kwargs.pop("prompt", None)
        backward_scale = kwargs.pop("backward_scale", 1)
        forward_scale = kwargs.pop("forward_scale", 1)
        scaling_steps = kwargs.pop("scaling_steps", range(len(timesteps))[1:])
        vbench_dimension_list = kwargs.pop("vbench_dimension_list", ["overall_consistency"])
        vbench_gpus = kwargs.pop("vbench_gpus", [4,5,6,7])


        vbench_gpus = vbench_gpus[:num_subtree]

        # cond ref arguments
        masks = kwargs.pop("masks")
        masked_ref = kwargs.pop("masked_ref")
        kwargs.pop("sigma_min")

        # oscillation guidance
        text_osci = kwargs.pop("text_osci", False)
        image_osci = kwargs.pop("image_osci", False)
        scale_temporal_osci = kwargs.pop("scale_temporal_osci", False)

        prompt = [prompt[0]]
        prompt_short = sanitize_filename(prompt[0])
        save_dir = f'temp/{prompt_short}'
        os.makedirs(save_dir, exist_ok=True)

        def save_video(tensor, filename):
            # tensor shape: [B, C, T, H, W]
            # Normalize to [0, 255]
            video = ((tensor + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            if video.shape[2] == 1:  # Single frame - save as image
                # Save middle frame for evaluation
                frame = video[0, :, 0]  # [H, W, C]
                image_path = filename.replace('.mp4', '.png')
                torchvision.io.write_png(frame.cpu(), image_path)
                return image_path
            else:
                # Also save full video
                video = video[0].permute(1, 2, 3, 0)[20:].contiguous()  # [B, T, H, W, C]
                torchvision.io.write_video(filename, video.cpu(), fps=8)
                return filename

        guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            # timesteps
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            b, c, t, w, h = masked_ref.size()
            cond = torch.cat((masks, masked_ref), dim=1)
            cond = pack(cond)
            kwargs["cond"] = torch.cat([cond, cond, torch.zeros_like(cond)], dim=0)

            # forward preparation
            cond_x = img[: len(img) // 3]

            noise_shape = cond_x.shape
            img = torch.cat([cond_x, cond_x, cond_x], dim=0)


            if i not in scaling_steps or not do_inference_scaling:
                # forward
                pred = model(
                    img=img,
                    **kwargs,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )

                # prepare guidance
                text_gs = get_oscillation_gs(guidance, i) if text_osci else guidance
                image_gs = get_oscillation_gs(guidance_img, i) if image_osci else guidance_img
                cond, uncond, uncond_2 = pred.chunk(3, dim=0)
                if image_gs > 1.0 and scale_temporal_osci:
                    # image_gs decrease with each denoising step
                    step_upper_image_gs = torch.linspace(image_gs, 1.0, len(timesteps))[i]
                    # image_gs increase along the temporal axis of the latent video
                    image_gs = torch.linspace(1.0, step_upper_image_gs, t)[None, None, :, None, None].repeat(b, c, 1, h, w)
                    image_gs = pack(image_gs).to(cond.device, cond.dtype)

                # update
                pred = uncond_2 + image_gs * (uncond - uncond_2) + text_gs * (cond - uncond)
                pred = torch.cat([pred, pred, pred], dim=0)

                img = img + (t_prev - t_curr) * pred
            else:
                subtrees = []
                subtree_onesteps = []
                
                # Generate multiple subtrees
                for subtree_idx in range(num_subtree):
                    noise = torch.randn(noise_shape, device=cond_x.device, dtype=cond_x.dtype)
                    zeros = torch.zeros(noise_shape, device=cond_x.device, dtype=cond_x.dtype)
                    noise = torch.cat([noise, zeros, zeros], dim=0)
                    subtree = img - (timesteps[i+1] - t_curr) * forward_scale * noise
                    t_subtree = t_curr
                    t_subtree_prev = t_subtree + (timesteps[i+1] - t_curr) * backward_scale
                    t_subtree_vec = torch.full((img.shape[0],), t_subtree, dtype=cond.dtype, device=cond.device)
                    subtree_noise_pred = model(
                        img=subtree,
                        **kwargs,
                        timesteps=t_subtree_vec,
                        guidance=guidance_vec,
                    )
                    # prepare guidance
                    text_gs = get_oscillation_gs(guidance, i) if text_osci else guidance
                    image_gs = get_oscillation_gs(guidance_img, i) if image_osci else guidance_img
                    cond, uncond, uncond_2 = subtree_noise_pred.chunk(3, dim=0)
                    if image_gs > 1.0 and scale_temporal_osci:
                        # image_gs decrease with each denoising step
                        step_upper_image_gs = torch.linspace(image_gs, 1.0, len(timesteps))[i]
                        # image_gs increase along the temporal axis of the latent video
                        image_gs = torch.linspace(1.0, step_upper_image_gs, t)[None, None, :, None, None].repeat(b, c, 1, h, w)
                        image_gs = pack(image_gs).to(cond.device, cond.dtype)

                    # update
                    subtree_noise_pred = uncond_2 + image_gs * (uncond - uncond_2) + text_gs * (cond - uncond)
                    subtree_noise_pred = torch.cat([subtree_noise_pred, subtree_noise_pred, subtree_noise_pred], dim=0)

                    subtree = subtree + (t_subtree_prev - t_subtree) * subtree_noise_pred

                    # Next timestep prediction
                    t_subtree_prev_vec = torch.full((img.shape[0],), t_subtree_prev, dtype=cond.dtype, device=cond.device)
                    subtree_onestep_noise_pred = model(
                        img=subtree,
                        **kwargs,
                        timesteps=t_subtree_prev_vec,
                        guidance=guidance_vec,
                    )
                    # prepare guidance
                    text_gs = get_oscillation_gs(guidance, i) if text_osci else guidance
                    image_gs = get_oscillation_gs(guidance_img, i) if image_osci else guidance_img
                    cond, uncond, uncond_2 = subtree_noise_pred.chunk(3, dim=0)
                    if image_gs > 1.0 and scale_temporal_osci:
                        # image_gs decrease with each denoising step
                        step_upper_image_gs = torch.linspace(image_gs, 1.0, len(timesteps))[i]
                        # image_gs increase along the temporal axis of the latent video
                        image_gs = torch.linspace(1.0, step_upper_image_gs, t)[None, None, :, None, None].repeat(b, c, 1, h, w)
                        image_gs = pack(image_gs).to(cond.device, cond.dtype)

                    # update
                    subtree_noise_pred = uncond_2 + image_gs * (uncond - uncond_2) + text_gs * (cond - uncond)
                    subtree_noise_pred = torch.cat([subtree_noise_pred, subtree_noise_pred, subtree_noise_pred], dim=0)

                    # img = img + (t_prev - t_curr) * pred
                    subtree_onestep = subtree + (timesteps[-1] - t_subtree_prev) * subtree_onestep_noise_pred
                    # Save subtree results
                    if model_ae is not None:
                        unpacked_subtree_onestep = unpack(subtree_onestep, height, width, num_frames, patch_size)[: len(img) // 3]
                        decoded_onestep = model_ae.decode(unpacked_subtree_onestep)
                        # Now returns path to middle frame image for evaluation
                        os.makedirs(f"{save_dir}/{i}_subtree", exist_ok=True)
                        decoded_onestep_path = save_video(decoded_onestep, f"{save_dir}/{i}_subtree/{subtree_idx}.mp4")
                    
                    # Store results
                    subtrees.append(subtree)
                    subtree_onesteps.append(decoded_onestep_path)
                
                dist.barrier()
                if dist.get_rank() == 0:
                    videos_path = f"{save_dir}/{i}_subtree"
                    output_path = f"{save_dir}/{i}_subtree"
                    
                    prompt_file = "temp/prompt.json" # hard coded for now
                    with open(prompt_file, "w") as fp:
                        prompt_json = {f"{index}.mp4": prompt[0] for index in range(num_subtree)}
                        json.dump(prompt_json, fp)
                    
                    python_path = os.path.dirname(sys.executable)
                    minimal_env = {
                        "PATH": f"{python_path}:/usr/local/bin:/usr/bin:/bin",
                        "CUDA_VISIBLE_DEVICES": ",".join([str(item) for item in vbench_gpus])
                    }
                    cmd_args = [
                        'vbench', 
                        'evaluate',
                        '--dimension', 
                        ' '.join(vbench_dimension_list),
                        '--videos_path', 
                        videos_path,
                        '--mode',
                        'custom_input',
                        '--prompt_file', 
                        prompt_file,
                        '--output_path', 
                        output_path,
                        '--ngpus',
                        str(len(vbench_gpus))
                    ]
                    subprocess.run(cmd_args, check=True, env=minimal_env)

                dist.barrier()
                # Find the latest evaluation results file
                subtree_dir = f"{save_dir}/{i}_subtree"
                eval_files = [f for f in os.listdir(subtree_dir) if f.startswith('results_') and f.endswith('_eval_results.json')]
                if not eval_files:
                    raise FileNotFoundError(f"No evaluation results found in {subtree_dir}")
                # Sort by timestamp in filename to get the latest one
                latest_eval_file = sorted(eval_files)[-1]
                eval_results_path = os.path.join(subtree_dir, latest_eval_file)
                
                with open(eval_results_path, 'r') as f:
                    eval_results = json.load(f)
                    best_idx = find_highest_score_video(eval_results)
                img = subtrees[best_idx]

        img = img[: len(img) // 3]

        return img

    def prepare_guidance(
        self, text: list[str], optional_models: dict[str, nn.Module], device: torch.device, dtype: torch.dtype, **kwargs
    ) -> tuple[list[str], dict[str, Tensor]]:
        ret = {}

        neg = kwargs.get("neg", None)
        ret["guidance_img"] = kwargs.pop("guidance_img")

        # text
        if neg is None:
            neg = [""] * len(text)
        text = text + neg + neg
        return text, ret
    

class DistilledDenoiser(Denoiser):
    def denoise(self, model: MMDiTModel, **kwargs) -> Tensor:
        img = kwargs.pop("img")
        timesteps = kwargs.pop("timesteps")
        guidance = kwargs.pop("guidance")

        guidance_vec = torch.full(
            (img.shape[0],), guidance, device=img.device, dtype=img.dtype
        )
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            # timesteps
            t_vec = torch.full(
                (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
            )
            # forward
            pred = model(
                img=img,
                **kwargs,
                timesteps=t_vec,
                guidance=guidance_vec,
            )
            # update
            img = img + (t_prev - t_curr) * pred
        return img

    def prepare_guidance(
        self,
        text: list[str],
        optional_models: dict[str, nn.Module],
        device: torch.device,
        dtype: torch.dtype,
        **kwargs,
    ) -> tuple[list[str], dict[str, Tensor]]:
        return text, {}


SamplingMethodDict = {
    SamplingMethod.I2V: I2VDenoiser(),
    SamplingMethod.I2VINFERENCESCALING: I2VScalingDenoiser(),
    SamplingMethod.DISTILLED: DistilledDenoiser(),
}


# ======================================================
# Timesteps
# ======================================================


def time_shift(alpha: float, t: Tensor) -> Tensor:
    return alpha * t / (1 + (alpha - 1) * t)


def get_res_lin_function(
    x1: float = 256, y1: float = 1, x2: float = 4096, y2: float = 3
) -> callable:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    num_frames: int,
    shift_alpha: float | None = None,
    base_shift: float = 1,
    max_shift: float = 3,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        if shift_alpha is None:
            # estimate mu based on linear estimation between two points
            # spatial scale
            shift_alpha = get_res_lin_function(y1=base_shift, y2=max_shift)(
                image_seq_len
            )
            # temporal scale
            shift_alpha *= math.sqrt(num_frames)
        # calculate shifted timesteps
        timesteps = time_shift(shift_alpha, timesteps)

    return timesteps.tolist()


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    patch_size: int = 2,
    channel: int = 16,
) -> Tensor:
    """
    Generate a noise tensor.

    Args:
        num_samples (int): Number of samples.
        height (int): Height of the noise tensor.
        width (int): Width of the noise tensor.
        num_frames (int): Number of frames.
        device (torch.device): Device to put the noise tensor on.
        dtype (torch.dtype): Data type of the noise tensor.
        seed (int): Seed for the random number generator.

    Returns:
        Tensor: The noise tensor.
    """
    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))
    return torch.randn(
        num_samples,
        channel,
        num_frames,
        # allow for packing
        patch_size * math.ceil(height / D),
        patch_size * math.ceil(width / D),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def pack(x: Tensor, patch_size: int = 2) -> Tensor:
    return rearrange(
        x, "b c t (h ph) (w pw) -> b (t h w) (c ph pw)", ph=patch_size, pw=patch_size
    )


def unpack(
    x: Tensor, height: int, width: int, num_frames: int, patch_size: int = 2
) -> Tensor:
    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))
    return rearrange(
        x,
        "b (t h w) (c ph pw) -> b c t (h ph) (w pw)",
        h=math.ceil(height / D),
        w=math.ceil(width / D),
        t=num_frames,
        ph=patch_size,
        pw=patch_size,
    )


# ======================================================
# Prepare
# ======================================================


def prepare(
    t5,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    seq_align: int = 1,
    patch_size: int = 2,
) -> dict[str, Tensor]:
    """
    Prepare the input for the model.

    Args:
        t5 (HFEmbedder): The T5 model.
        clip (HFEmbedder): The CLIP model.
        img (Tensor): The image tensor.
        prompt (str | list[str]): The prompt(s).

    Returns:
        dict[str, Tensor]: The input dictionary.

        img_ids: used for positional embedding in T,H,W dimensions later
        text_ids: for positional embedding, but set to 0 for now since our text encoder already encodes positional information
    """
    bs, c, t, h, w = img.shape
    device, dtype = img.device, img.dtype
    if isinstance(prompt, str):
        prompt = [prompt]
    if bs != len(prompt):
        bs = len(prompt)

    img = rearrange(
        img, "b c t (h ph) (w pw) -> b (t h w) (c ph pw)", ph=patch_size, pw=patch_size
    )
    if img.shape[0] != bs:
        img = repeat(img, "b ... -> (repeat b) ...", repeat=bs // img.shape[0])

    img_ids = torch.zeros(t, h // patch_size, w // patch_size, 3)
    img_ids[..., 0] = img_ids[..., 0] + torch.arange(t)[:, None, None]
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // patch_size)[None, :, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // patch_size)[None, None, :]
    img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)

    # Encode the tokenized prompts
    txt = t5(prompt, added_tokens=img_ids.shape[1], seq_align=seq_align)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(device, dtype),
        "txt": txt.to(device, dtype),
        "txt_ids": txt_ids.to(device, dtype),
        "y_vec": vec.to(device, dtype),
    }


def prepare_ids(
    img: Tensor,
    t5_embedding: Tensor,
    clip_embedding: Tensor,
) -> dict[str, Tensor]:
    """
    Prepare the input for the model.

    Args:
        img (Tensor): The image tensor.
        t5_embedding (Tensor): The T5 embedding.
        clip_embedding (Tensor): The CLIP embedding.

    Returns:
        dict[str, Tensor]: The input dictionary.

        img_ids: used for positional embedding in T,H,W dimensions later
        text_ids: for positional embedding, but set to 0 for now since our text encoder already encodes positional information
    """
    bs, c, t, h, w = img.shape
    device, dtype = img.device, img.dtype

    img = rearrange(img, "b c t (h ph) (w pw) -> b (t h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] != bs:
        img = repeat(img, "b ... -> (repeat b) ...", repeat=bs // img.shape[0])

    img_ids = torch.zeros(t, h // 2, w // 2, 3)
    img_ids[..., 0] = img_ids[..., 0] + torch.arange(t)[:, None, None]
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[None, :, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, None, :]
    img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)

    # Encode the tokenized prompts
    if t5_embedding.shape[0] == 1 and bs > 1:
        t5_embedding = repeat(t5_embedding, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, t5_embedding.shape[1], 3)

    if clip_embedding.shape[0] == 1 and bs > 1:
        clip_embedding = repeat(clip_embedding, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(device, dtype),
        "txt": t5_embedding.to(device, dtype),
        "txt_ids": txt_ids.to(device, dtype),
        "y_vec": clip_embedding.to(device, dtype),
    }


def prepare_models(
    cfg: Config,
    device: torch.device,
    dtype: torch.dtype,
    offload_model: bool = False,
) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, dict[str, nn.Module]]:
    """
    Prepare models for inference.

    Args:
        cfg (Config): The configuration object.
        device (torch.device): The device to use.
        dtype (torch.dtype): The data type to use.

    Returns:
        tuple[nn.Module, nn.Module, nn.Module, nn.Module, dict[str, nn.Module]]: The models. They are the diffusion model, the autoencoder model, the T5 model, the CLIP model, and the optional models.
    """
    model_device = (
        "cpu" if offload_model and cfg.get("img_flux", None) is not None else device
    )

    model = build_module(
        cfg.model, MODELS, device_map=model_device, torch_dtype=dtype
    ).eval()
    model_ae = build_module(
        cfg.ae, MODELS, device_map=model_device, torch_dtype=dtype
    ).eval()
    model_t5 = build_module(cfg.t5, MODELS, device_map=device, torch_dtype=dtype).eval()
    model_clip = build_module(
        cfg.clip, MODELS, device_map=device, torch_dtype=dtype
    ).eval()
    if cfg.get("pretrained_lora_path", None) is not None:
        model = PeftModel.from_pretrained(
            model, cfg.pretrained_lora_path, is_trainable=False
        )

    # optional models
    optional_models = {}
    if cfg.get("img_flux", None) is not None:
        model_img_flux = build_module(
            cfg.img_flux, MODELS, device_map=device, torch_dtype=dtype
        ).eval()
        model_ae_img_flux = build_module(
            cfg.img_flux_ae, MODELS, device_map=device, torch_dtype=dtype
        ).eval()
        optional_models["img_flux"] = model_img_flux
        optional_models["img_flux_ae"] = model_ae_img_flux

    return model, model_ae, model_t5, model_clip, optional_models


def prepare_api(
    model: nn.Module,
    model_ae: nn.Module,
    model_t5: nn.Module,
    model_clip: nn.Module,
    optional_models: dict[str, nn.Module],
) -> callable:
    """
    Prepare the API function for inference.

    Args:
        model (nn.Module): The diffusion model.
        model_ae (nn.Module): The autoencoder model.
        model_t5 (nn.Module): The T5 model.
        model_clip (nn.Module): The CLIP model.

    Returns:
        callable: The API function for inference.
    """

    @torch.inference_mode()
    def api_fn(
        opt: SamplingOption,
        cond_type: str = "t2v",
        seed: int = None,
        sigma_min: float = 1e-5,
        text: list[str] = None,
        neg: list[str] = None,
        patch_size: int = 2,
        channel: int = 16,
        **kwargs,
    ):
        """
        The API function for inference.

        Args:
            opt (SamplingOption): The sampling options.
            text (list[str], optional): The text prompts. Defaults to None.
            neg (list[str], optional): The negative text prompts. Defaults to None.

        Returns:
            torch.Tensor: The generated images.
        """
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # passing seed will overwrite opt seed
        if seed is None:
            # random seed if not provided
            seed = opt.seed if opt.seed is not None else random.randint(0, 2**32 - 1)
        if opt.is_causal_vae:
            num_frames = (
                1
                if opt.num_frames == 1
                else (opt.num_frames - 1) // opt.temporal_reduction + 1
            )
        else:
            num_frames = (
                1 if opt.num_frames == 1 else opt.num_frames // opt.temporal_reduction
            )

        z = get_noise(
            len(text),
            opt.height,
            opt.width,
            num_frames,
            device,
            dtype,
            seed,
            patch_size=patch_size,
            channel=channel // (patch_size**2),
        )
        denoiser = SamplingMethodDict[opt.method]

        # i2v reference conditions
        references = [None] * len(text)
        if cond_type != "t2v" and "ref" in kwargs:
            reference_path_list = kwargs.pop("ref")
            references = collect_references_batch(
                reference_path_list,
                cond_type,
                model_ae,
                (opt.height, opt.width),
                is_causal=opt.is_causal_vae,
            )
        elif cond_type != "t2v":
            print(
                "your csv file doesn't have a ref column or is not processed properly. will default to cond_type t2v!"
            )
            cond_type = "t2v"

        # timestep editing
        timesteps = get_schedule(
            opt.num_steps,
            (z.shape[-1] * z.shape[-2]) // patch_size**2,
            num_frames,
            shift=opt.shift,
            shift_alpha=opt.flow_shift,
        )

        # prepare classifier-free guidance data (method specific)
        text, additional_inp = denoiser.prepare_guidance(
            text=text,
            optional_models=optional_models,
            device=device,
            dtype=dtype,
            neg=neg,
            guidance_img=opt.guidance_img,
        )

        inp = prepare(model_t5, model_clip, z, prompt=text, patch_size=patch_size)
        inp.update(additional_inp)

        if opt.method in [SamplingMethod.I2V, SamplingMethod.I2VINFERENCESCALING]:
            # prepare references
            masks, masked_ref = prepare_inference_condition(
                z, cond_type, ref_list=references, causal=opt.is_causal_vae
            )
            inp["masks"] = masks
            inp["masked_ref"] = masked_ref
            inp["sigma_min"] = sigma_min

        x = denoiser.denoise(
            model,
            **inp,
            timesteps=timesteps,
            guidance=opt.guidance,
            text_osci=opt.text_osci,
            image_osci=opt.image_osci,
            scale_temporal_osci=(
                opt.scale_temporal_osci and "i2v" in cond_type
            ),  # don't use temporal osci for v2v or t2v
            flow_shift=opt.flow_shift,
            patch_size=patch_size,
            # inference scaling parameters
            do_inference_scaling=opt.do_inference_scaling,
            num_subtree=opt.num_subtree,
            model_ae=model_ae,
            height=opt.height,
            width=opt.width,
            backward_scale = opt.backward_scale,
            forward_scale = opt.forward_scale,
            scaling_steps = opt.scaling_steps,
            num_frames=num_frames,
            prompt=text,
            vbench_dimension_list=opt.vbench_dimension_list,
            vbench_gpus=opt.vbench_gpus
        )

        x = unpack(x, opt.height, opt.width, num_frames, patch_size=patch_size)

        # replace for image condition
        if cond_type == "i2v_head":
            x[0, :, :1] = references[0][0]
        elif cond_type == "i2v_tail":
            x[0, :, -1:] = references[0][0]
        elif cond_type == "i2v_loop":
            x[0, :, :1] = references[0][0]
            x[0, :, -1:] = references[0][1]

        x = model_ae.decode(x)
        x = x[:, :, : opt.num_frames]  # image

        # remove the duplicate frames
        if not opt.is_causal_vae:
            if cond_type == "i2v_head":
                pad_len = model_ae.compression[0] - 1
                x = x[:, :, pad_len:]
            elif cond_type == "i2v_tail":
                pad_len = model_ae.compression[0] - 1
                x = x[:, :, :-pad_len]
            elif cond_type == "i2v_loop":
                pad_len = model_ae.compression[0] - 1
                x = x[:, :, pad_len:-pad_len]

        return x

    return api_fn


def sanitize_filename(prompt):
    """Sanitize the prompt to create a valid filename."""
    # Remove or replace special characters
    invalid_chars = '<>:"/\\|?*\n\r\t'
    filename = prompt.strip()
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Replace multiple spaces/underscores with single underscore
    filename = '_'.join(filter(None, filename.split()))
    
    # Limit length and ensure it's not empty
    filename = filename[:30] if filename else "default"
    
    # Remove leading/trailing special characters
    filename = filename.strip('._-')
    
    return filename or "default"

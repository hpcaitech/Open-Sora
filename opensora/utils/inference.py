import copy
import os
import re
from enum import Enum

import torch
from torch import nn

from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size
from opensora.datasets.utils import read_from_path, rescale_image_by_path
from opensora.utils.logger import log_message
from opensora.utils.prompt_refine import refine_prompts


class SamplingMethod(Enum):
    I2V = "i2v"  # for open sora video generation
    DISTILLED = "distill"  # for flux image generation


def create_tmp_csv(save_dir: str, prompt: str, ref: str = None, create=True) -> str:
    """
    Create a temporary CSV file with the prompt text.

    Args:
        save_dir (str): The directory where the CSV file will be saved.
        prompt (str): The prompt text.

    Returns:
        str: The path to the temporary CSV file.
    """
    tmp_file = os.path.join(save_dir, "prompt.csv")
    if not create:
        return tmp_file
    with open(tmp_file, "w", encoding="utf-8") as f:
        if ref is not None:
            f.write(f'text,ref\n"{prompt}","{ref}"')
        else:
            f.write(f'text\n"{prompt}"')
    return tmp_file


def modify_option_to_t2i(sampling_option, distilled: bool = False, img_resolution: str = "1080px"):
    """
    Modify the sampling option to be used for text-to-image generation.
    """
    sampling_option_t2i = copy.copy(sampling_option)
    if distilled:
        sampling_option_t2i.method = SamplingMethod.DISTILLED
    sampling_option_t2i.num_frames = 1
    sampling_option_t2i.height, sampling_option_t2i.width = get_image_size(img_resolution, sampling_option.aspect_ratio)
    sampling_option_t2i.guidance = 4.0
    sampling_option_t2i.resized_resolution = sampling_option.resolution

    return sampling_option_t2i


def get_save_path_name(
    save_dir,
    sub_dir,
    save_prefix="",
    name=None,
    fallback_name=None,
    index=None,
    num_sample_pos=None,  # idx for prompt as path
    prompt_as_path=False,  # save sample with same name as prompt
    prompt=None,
):
    """
    Get the save path for the generated samples.
    """
    if prompt_as_path:  # for vbench
        cleaned_prompt = prompt.strip(".")
        fname = f"{cleaned_prompt}-{num_sample_pos}"
    else:
        if name is not None:
            fname = save_prefix + name
        else:
            fname = f"{save_prefix + fallback_name}_{index:04d}"
        if num_sample_pos > 0:
            fname += f"_{num_sample_pos}"

    return os.path.join(save_dir, sub_dir, fname)


def get_names_from_path(path):
    """
    Get the filename and extension from a path.

    Args:
        path (str): The path to the file.

    Returns:
        tuple[str, str]: The filename and the extension.
    """
    filename = os.path.basename(path)
    name, _ = os.path.splitext(filename)
    return name


def process_and_save(
    x: torch.Tensor,
    batch: dict,
    cfg: dict,
    sub_dir: str,
    generate_sampling_option,
    epoch: int,
    start_index: int,
    saving: bool = True,
):
    """
    Process the generated samples and save them to disk.
    """
    fallback_name = cfg.dataset.data_path.split("/")[-1].split(".")[0]
    prompt_as_path = cfg.get("prompt_as_path", False)
    fps_save = cfg.get("fps_save", 16)
    save_dir = cfg.save_dir

    names = batch["name"] if "name" in batch else [None] * len(x)
    indices = batch["index"] if "index" in batch else [None] * len(x)
    if "index" in batch:
        indices = [idx + start_index for idx in indices]
    prompts = batch["text"]

    ret_names = []
    is_image = generate_sampling_option.num_frames == 1
    for img, name, index, prompt in zip(x, names, indices, prompts):
        # == get save path ==
        save_path = get_save_path_name(
            save_dir,
            sub_dir,
            save_prefix=cfg.get("save_prefix", ""),
            name=name,
            fallback_name=fallback_name,
            index=index,
            num_sample_pos=epoch,
            prompt_as_path=prompt_as_path,
            prompt=prompt,
        )
        ret_name = get_names_from_path(save_path)
        ret_names.append(ret_name)

        if saving:
            # == write txt to disk ==
            with open(save_path + ".txt", "w", encoding="utf-8") as f:
                f.write(prompt)

            # == save samples ==
            save_sample(img, save_path=save_path, fps=fps_save)

            # == resize image for t2i2v ==
            if (
                cfg.get("use_t2i2v", False)
                and is_image
                and generate_sampling_option.resolution != generate_sampling_option.resized_resolution
            ):
                log_message("Rescaling image to %s...", generate_sampling_option.resized_resolution)
                height, width = get_image_size(
                    generate_sampling_option.resized_resolution, generate_sampling_option.aspect_ratio
                )
                rescale_image_by_path(save_path + ".png", width, height)

    return ret_names


def check_fps_added(sentence):
    """
    Check if the sentence ends with the FPS information.
    """
    pattern = r"\d+ FPS\.$"
    if re.search(pattern, sentence):
        return True
    return False


def ensure_sentence_ends_with_period(sentence: str):
    """
    Ensure that the sentence ends with a period.
    """
    sentence = sentence.strip()
    if not sentence.endswith("."):
        sentence += "."
    return sentence


def add_fps_info_to_text(text: list[str], fps: int = 16):
    """
    Add the FPS information to the text.
    """
    mod_text = []
    for item in text:
        item = ensure_sentence_ends_with_period(item)
        if not check_fps_added(item):
            item = item + f" {fps} FPS."
        mod_text.append(item)
    return mod_text


def add_motion_score_to_text(text, motion_score: int | str):
    """
    Add the motion score to the text.
    """
    if motion_score == "dynamic":
        ms = refine_prompts(text, type="motion_score")
        return [f"{t} {ms[i]}." for i, t in enumerate(text)]
    else:
        return [f"{t} {motion_score} motion score." for t in text]


def add_noise_to_ref(masked_ref: torch.Tensor, masks: torch.Tensor, t: float, sigma_min: float = 1e-5):
    z_1 = torch.randn_like(masked_ref)
    z_noisy = (1 - (1 - sigma_min) * t) * masked_ref + t * z_1
    return masks * z_noisy


def collect_references_batch(
    reference_paths: list[str],
    cond_type: str,
    model_ae: nn.Module,
    image_size: tuple[int, int],
    is_causal=False,
):
    refs_x = []  # refs_x: [batch, ref_num, C, T, H, W]
    device = next(model_ae.parameters()).device
    dtype = next(model_ae.parameters()).dtype
    for reference_path in reference_paths:
        if reference_path == "":
            refs_x.append(None)
            continue
        ref_path = reference_path.split(";")
        ref = []

        if "v2v" in cond_type:
            r = read_from_path(ref_path[0], image_size, transform_name="resize_crop")  # size [C, T, H, W]
            actual_t = r.size(1)
            target_t = (
                64 if (actual_t >= 64 and "easy" in cond_type) else 32
            )  # if reference not long enough, default to shorter ref
            if is_causal:
                target_t += 1
            assert actual_t >= target_t, f"need at least {target_t} reference frames for v2v generation"
            if "head" in cond_type:  # v2v head
                r = r[:, :target_t]
            elif "tail" in cond_type:  # v2v tail
                r = r[:, -target_t:]
            else:
                raise NotImplementedError
            r_x = model_ae.encode(r.unsqueeze(0).to(device, dtype))
            r_x = r_x.squeeze(0)  # size [C, T, H, W]
            ref.append(r_x)
        elif cond_type == "i2v_head":  # take the 1st frame from first ref_path
            r = read_from_path(ref_path[0], image_size, transform_name="resize_crop")  # size [C, T, H, W]
            r = r[:, :1]
            r_x = model_ae.encode(r.unsqueeze(0).to(device, dtype))
            r_x = r_x.squeeze(0)  # size [C, T, H, W]
            ref.append(r_x)
        elif cond_type == "i2v_tail":  # take the last frame from last ref_path
            r = read_from_path(ref_path[-1], image_size, transform_name="resize_crop")  # size [C, T, H, W]
            r = r[:, -1:]
            r_x = model_ae.encode(r.unsqueeze(0).to(device, dtype))
            r_x = r_x.squeeze(0)  # size [C, T, H, W]
            ref.append(r_x)
        elif cond_type == "i2v_loop":
            # first frame
            r_head = read_from_path(ref_path[0], image_size, transform_name="resize_crop")  # size [C, T, H, W]
            r_head = r_head[:, :1]
            r_x_head = model_ae.encode(r_head.unsqueeze(0).to(device, dtype))
            r_x_head = r_x_head.squeeze(0)  # size [C, T, H, W]
            ref.append(r_x_head)
            # last frame
            r_tail = read_from_path(ref_path[-1], image_size, transform_name="resize_crop")  # size [C, T, H, W]
            r_tail = r_tail[:, -1:]
            r_x_tail = model_ae.encode(r_tail.unsqueeze(0).to(device, dtype))
            r_x_tail = r_x_tail.squeeze(0)  # size [C, T, H, W]
            ref.append(r_x_tail)
        else:
            raise NotImplementedError(f"Unknown condition type {cond_type}")

        refs_x.append(ref)
    return refs_x


def prepare_inference_condition(
    z: torch.Tensor,
    mask_cond: str,
    ref_list: list[list[torch.Tensor]] = None,
    causal: bool = True,
) -> torch.Tensor:
    """
    Prepare the visual condition for the model, using causal vae.

    Args:
        z (torch.Tensor): The latent noise tensor, of shape [B, C, T, H, W]
        mask_cond (dict): The condition configuration.
        ref_list: list of lists of media (image/video) for i2v and v2v condition, of shape [C, T', H, W]; len(ref_list)==B; ref_list[i] is the list of media for the generation in batch idx i, we use a list of media for each batch item so that it can have multiple references. For example, ref_list[i] could be [ref_image_1, ref_image_2] for i2v_loop condition.

    Returns:
        torch.Tensor: The visual condition tensor.
    """
    # x has shape [b, c, t, h, w], where b is the batch size
    B, C, T, H, W = z.shape

    masks = torch.zeros(B, 1, T, H, W)
    masked_z = torch.zeros(B, C, T, H, W)

    if ref_list is None:
        assert mask_cond == "t2v", f"reference is required for {mask_cond}"

    for i in range(B):
        ref = ref_list[i]

        # warning message
        if ref is None and mask_cond != "t2v":
            print("no reference found. will default to cond_type t2v!")

        if ref is not None and T > 1:  # video
            # Apply the selected mask condition directly on the masks tensor
            if mask_cond == "i2v_head":  # equivalent to masking the first timestep
                masks[i, :, 0, :, :] = 1
                masked_z[i, :, 0, :, :] = ref[0][:, 0, :, :]
            elif mask_cond == "i2v_tail":  # mask the last timestep
                masks[i, :, -1, :, :] = 1
                masked_z[i, :, -1, :, :] = ref[-1][:, -1, :, :]
            elif mask_cond == "v2v_head":
                k = 8 + int(causal)
                masks[i, :, :k, :, :] = 1
                masked_z[i, :, :k, :, :] = ref[0][:, :k, :, :]
            elif mask_cond == "v2v_tail":
                k = 8 + int(causal)
                masks[i, :, -k:, :, :] = 1
                masked_z[i, :, -k:, :, :] = ref[0][:, -k:, :, :]
            elif mask_cond == "v2v_head_easy":
                k = 16 + int(causal)
                masks[i, :, :k, :, :] = 1
                masked_z[i, :, :k, :, :] = ref[0][:, :k, :, :]
            elif mask_cond == "v2v_tail_easy":
                k = 16 + int(causal)
                masks[i, :, -k:, :, :] = 1
                masked_z[i, :, -k:, :, :] = ref[0][:, -k:, :, :]
            elif mask_cond == "i2v_loop":  # mask first and last timesteps
                masks[i, :, 0, :, :] = 1
                masks[i, :, -1, :, :] = 1
                masked_z[i, :, 0, :, :] = ref[0][:, 0, :, :]
                masked_z[i, :, -1, :, :] = ref[-1][:, -1, :, :]  # last frame of last referenced content
            else:
                # "t2v" is the fallback case where no specific condition is specified
                assert mask_cond == "t2v", f"Unknown mask condition {mask_cond}"

    masks = masks.to(z.device, z.dtype)
    masked_z = masked_z.to(z.device, z.dtype)
    return masks, masked_z

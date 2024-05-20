import json
import os
import re

import torch

from opensora.datasets import IMG_FPS
from opensora.datasets.utils import read_from_path


def prepare_multi_resolution_info(info_type, batch_size, image_size, num_frames, fps, device, dtype):
    if info_type is None:
        return dict()
    elif info_type == "PixArtMS":
        hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(batch_size, 1)
        ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(batch_size, 1)
        return dict(ar=ar, hw=hw)
    elif info_type in ["STDiT2", "OpenSora"]:
        fps = fps if num_frames > 1 else IMG_FPS
        fps = torch.tensor([fps], device=device, dtype=dtype).repeat(batch_size)
        height = torch.tensor([image_size[0]], device=device, dtype=dtype).repeat(batch_size)
        width = torch.tensor([image_size[1]], device=device, dtype=dtype).repeat(batch_size)
        num_frames = torch.tensor([num_frames], device=device, dtype=dtype).repeat(batch_size)
        ar = torch.tensor([image_size[0] / image_size[1]], device=device, dtype=dtype).repeat(batch_size)
        return dict(height=height, width=width, num_frames=num_frames, ar=ar, fps=fps)
    else:
        raise NotImplementedError


def load_prompts(prompt_path, start_idx=None, end_idx=None):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    prompts = prompts[start_idx:end_idx]
    return prompts


def get_save_path_name(
    save_dir,
    sample_name=None,  # prefix
    sample_idx=None,  # sample index
    prompt=None,  # used prompt
    prompt_as_path=False,  # use prompt as path
    num_sample=1,  # number of samples to generate for one prompt
    k=None,  # kth sample
):
    if sample_name is None:
        sample_name = "" if prompt_as_path else "sample"
    sample_name_suffix = prompt if prompt_as_path else f"_{sample_idx}"
    save_path = os.path.join(save_dir, f"{sample_name}{sample_name_suffix}")
    if num_sample != 1:
        save_path = f"{save_path}-{k}"
    return save_path


def extract_json_from_prompts(prompts, reference, mask_strategy):
    ret_prompts = []
    for i, prompt in enumerate(prompts):
        parts = re.split(r"(?=[{\[])", prompt)
        assert len(parts) <= 2, f"Invalid prompt: {prompt}"
        ret_prompts.append(parts[0])
        if len(parts) > 1:
            additional_info = json.loads(parts[1])
            for key in additional_info:
                assert key in ["reference_path", "mask_strategy"], f"Invalid key: {key}"
                if key == "reference_path":
                    reference[i] = additional_info[key]
                elif key == "mask_strategy":
                    mask_strategy[i] = additional_info[key]
    return ret_prompts, reference, mask_strategy


def collect_references_batch(reference_paths, vae, image_size):
    refs_x = []  # refs_x: [batch, ref_num, C, T, H, W]
    for reference_path in reference_paths:
        if reference_path == "":
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
    return refs_x


def extract_prompts_loop(prompts, num_loop):
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
            prompt = text_list[num_loop]
        ret_prompts.append(prompt)
    return ret_prompts


MASK_DEFAULT = ["0", "0", "0", "0", "1", "0"]


def parse_mask_strategy(mask_strategy):
    mask_batch = []
    if mask_strategy == "" or mask_strategy is None:
        return mask_batch

    mask_strategy = mask_strategy.split(";")
    for mask in mask_strategy:
        mask_group = mask.split(",")
        num_group = len(mask_group)
        assert num_group >= 1 and num_group <= 6, f"Invalid mask strategy: {mask}"
        mask_group.extend(MASK_DEFAULT[num_group:])
        for i in range(5):
            mask_group[i] = int(mask_group[i])
        mask_group[5] = float(mask_group[5])
        mask_batch.append(mask_group)
    return mask_batch


def find_nearest_point(value, point, max_value):
    t = value // point
    if value % point > point / 2 and t < max_value // point - 1:
        t += 1
    return t * point


def apply_mask_strategy(z, refs_x, mask_strategys, loop_i, align=None):
    masks = []
    no_mask = True
    for i, mask_strategy in enumerate(mask_strategys):
        no_mask = False
        mask = torch.ones(z.shape[2], dtype=torch.float, device=z.device)
        mask_strategy = parse_mask_strategy(mask_strategy)
        for mst in mask_strategy:
            loop_id, m_id, m_ref_start, m_target_start, m_length, edit_ratio = mst
            if loop_id != loop_i:
                continue
            ref = refs_x[i][m_id]

            if m_ref_start < 0:
                # ref: [C, T, H, W]
                m_ref_start = ref.shape[1] + m_ref_start
            if m_target_start < 0:
                # z: [B, C, T, H, W]
                m_target_start = z.shape[2] + m_target_start
            if align is not None:
                m_ref_start = find_nearest_point(m_ref_start, align, ref.shape[1])
                m_target_start = find_nearest_point(m_target_start, align, z.shape[2])
            m_length = min(m_length, z.shape[2] - m_target_start, ref.shape[1] - m_ref_start)
            z[i, :, m_target_start : m_target_start + m_length] = ref[:, m_ref_start : m_ref_start + m_length]
            mask[m_target_start : m_target_start + m_length] = edit_ratio
        masks.append(mask)
    if no_mask:
        return None
    masks = torch.stack(masks)
    return masks


def append_generated(vae, generated_video, refs_x, mask_strategy, loop_i, condition_frame_length):
    ref_x = vae.encode(generated_video)
    for j, refs in enumerate(refs_x):
        if refs is None:
            refs_x[j] = [ref_x[j]]
        else:
            refs.append(ref_x[j])
        if mask_strategy[j] is None:
            mask_strategy[j] = ""
        else:
            mask_strategy[j] += ";"
        mask_strategy[j] += f"{loop_i},{len(refs)-1},-{condition_frame_length},0,{condition_frame_length}"
    return refs_x, mask_strategy
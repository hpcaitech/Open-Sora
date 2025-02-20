import base64
import json
import os
import re
import shutil
import tempfile
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

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


def load_prompts(prompt_path, start_idx=None, end_idx=None, csv_ref_column_name=None):
    if prompt_path.endswith(".txt"):
        with open(prompt_path, "r") as f:
            prompts = [line.strip() for line in f.readlines()]
    elif prompt_path.endswith(".csv"):
        df = pd.read_csv(prompt_path)
        prompts = df["text"].tolist()
        if csv_ref_column_name is not None:
            assert (
                csv_ref_column_name in df
            ), f"column {csv_ref_column_name} for reference paths not found in {prompt_path}"
            reference_paths = df[csv_ref_column_name]
            return (prompts, reference_paths)
    else:
        raise ValueError
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
    sample_name_suffix = prompt if prompt_as_path else f"_{sample_idx:04d}"
    save_path = os.path.join(save_dir, f"{sample_name}{sample_name_suffix}")
    if num_sample != 1:
        save_path = f"{save_path}-{k}"
    return save_path


def transform_aes(aes):
    # < 4 filter out
    if aes < 4:
        return "terrible"
    elif aes < 4.5:
        return "very poor"
    elif aes < 5:
        return "poor"
    elif aes < 5.5:
        return "fair"
    elif aes < 6:
        return "good"
    elif aes < 6.5:
        return "very good"
    else:
        return "excellent"


def transform_motion(motion):
    # < 0.3 filter out
    if motion < 0.5:
        return "very low"
    elif motion < 2:
        return "low"
    elif motion < 5:
        return "fair"
    elif motion < 10:
        return "high"
    elif motion < 20:
        return "very high"
    else:
        return "extremely high"


def append_score_to_prompts(prompts, aes=None, flow=None, camera_motion=None):
    new_prompts = []
    for prompt in prompts:
        new_prompt = prompt
        if aes is not None and "aesthetic score is" not in prompt:
            try:
                aes = float(aes)
                aes = transform_aes(aes)
            except ValueError:
                pass  # already in text format
            new_prompt = f"{new_prompt} The aesthetic score is {aes}."

        if flow is not None and "motion strength is" not in prompt:
            try:
                flow = float(flow)
                flow = transform_motion(flow)
            except ValueError:
                pass  # already in text format
            new_prompt = f"{new_prompt} The motion strength is {flow}."
        if camera_motion is not None and "camera motion:" not in prompt:
            new_prompt = f"{new_prompt} camera motion: {camera_motion}."
        new_prompts.append(new_prompt)
    print("processed prompt:\n", new_prompts)
    return new_prompts


def extract_json_from_prompts(prompts, reference, mask_strategy):
    ret_prompts = []
    for i, prompt in enumerate(prompts):
        parts = re.split(r"(?=[{])", prompt)
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


def extract_json_from_prompts_new(prompts_with_json):
    prompts = []
    reference = []
    mask_strategy = []
    for i, prompt in enumerate(prompts_with_json):
        parts = re.split(r"(?=[{])", prompt)
        assert len(parts) <= 2, f"Invalid prompt: {prompt}"
        prompts.append(parts[0])
        if len(parts) > 1:
            additional_info = json.loads(parts[1])
            for key in additional_info:
                assert key in ["reference_path", "mask_strategy"], f"Invalid key: {key}"
                if key == "reference_path":
                    reference[i] = additional_info[key]
                elif key == "mask_strategy":
                    mask_strategy[i] = additional_info[key]
    return prompts, reference, mask_strategy


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

            # need to ensure r has length accepted by vae
            actual_t = r.size(1)
            if vae.micro_frame_size is None:
                target_t = (actual_t - 1) // 4 * 4 + 1
            elif not vae.temporal_overlap:
                target_t = actual_t // vae.micro_frame_size * vae.micro_frame_size
            else:
                target_t = (actual_t - 1) // (vae.micro_frame_size - 1) * (vae.micro_frame_size - 1) + 1
            r = r[:, :target_t]

            r_x = vae.encode(r.unsqueeze(0).to(vae.device, vae.dtype))
            r_x = r_x.squeeze(0)
            ref.append(r_x)
        refs_x.append(ref)
    return refs_x


def extract_images_from_ref_paths(reference_paths, image_size):
    refs_images = []  # refs_images: [batch, ref_num, C, T, H, W]
    for reference_path in reference_paths:
        if reference_path == "":
            refs_images.append([])
            continue
        ref_path = reference_path.split(";")
        ref = []
        for r_path in ref_path:
            r = read_from_path(r_path, image_size, transform_name="resize_crop")
            ref.append(r.squeeze(1))
        refs_images.append(ref)
    return refs_images


def extract_prompts_loop(prompts, num_loop):
    ret_prompts = []
    for prompt in prompts:
        if prompt.startswith("|0|"):
            prompt_list = prompt.split("|")[1:]
            text_list = []
            for i in range(0, len(prompt_list), 2):
                start_loop = int(prompt_list[i])
                text = prompt_list[i + 1]
                end_loop = int(prompt_list[i + 2]) if i + 2 < len(prompt_list) else num_loop + 1
                text_list.extend([text] * (end_loop - start_loop))
            prompt = text_list[num_loop]
        ret_prompts.append(prompt)
    return ret_prompts


def split_prompt(prompt_text):
    if prompt_text.startswith("|0|"):
        # this is for prompts which look like
        # |0| a beautiful day |1| a sunny day |2| a rainy day
        # we want to parse it into a list of prompts with the loop index
        prompt_list = prompt_text.split("|")[1:]
        text_list = []
        loop_idx = []
        for i in range(0, len(prompt_list), 2):
            start_loop = int(prompt_list[i])
            text = prompt_list[i + 1].strip()
            text_list.append(text)
            loop_idx.append(start_loop)
        return text_list, loop_idx
    else:
        return [prompt_text], None


def merge_prompt(text_list, loop_idx_list=None):
    if loop_idx_list is None:
        return text_list[0]
    else:
        prompt = ""
        for i, text in enumerate(text_list):
            prompt += f"|{loop_idx_list[i]}|{text}"
        return prompt


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


def append_generated(
    vae, generated_video, refs_x, mask_strategy, loop_i, condition_frame_length, condition_frame_edit, is_latent=False
):
    ref_x = vae.encode(generated_video) if not is_latent else generated_video

    for j, refs in enumerate(refs_x):
        if refs is None:
            refs_x[j] = [ref_x[j]]
        else:
            refs.append(ref_x[j])
        if mask_strategy[j] is None or mask_strategy[j] == "":
            mask_strategy[j] = ""
        else:
            mask_strategy[j] += ";"
        mask_strategy[
            j
        ] += f"{loop_i},{len(refs)-1},-{condition_frame_length},0,{condition_frame_length},{condition_frame_edit}"
    return refs_x, mask_strategy


def dframe_to_frame(num):
    assert num % 5 == 0, f"Invalid num: {num}"
    return num // 5 * 17


OPENAI_CLIENT = None
REFINE_PROMPTS = None
# REFINE_PROMPTS_PATH = "assets/texts/t2v_pllava.txt"
REFINE_PROMPTS_PATH = "assets/texts/t2v_demo.txt"
REFINE_PROMPTS_TEMPLATE = """
You need to refine user's input prompt. The user's input prompt is used for video generation task. You need to refine the user's prompt to make it more suitable for the task. Here are some examples of refined prompts:
{}

The refined prompt should pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. The video should not have any scene transitions and must be on the same scene. The refined prompt should be in English.
"""
RANDOM_PROMPTS = None
RANDOM_PROMPTS_TEMPLATE = """
You need to generate one input prompt for video generation task. The prompt should be suitable for the task. Here are some examples of refined prompts:
{}

The prompt should pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. The video should not have any scene transitions and must be on the same scene. The prompt should be in English.
"""


def get_openai_response(sys_prompt, usr_prompt, model="gpt-4o"):
    global OPENAI_CLIENT
    if OPENAI_CLIENT is None:
        from openai import OpenAI

        OPENAI_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    completion = OPENAI_CLIENT.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": sys_prompt,
            },  # <-- This is the system message that provides context to the model
            {
                "role": "user",
                "content": usr_prompt,
            },  # <-- This is the user message for which the model will generate a response
        ],
    )

    return completion.choices[0].message.content


def get_random_prompt_by_openai():
    global RANDOM_PROMPTS
    if RANDOM_PROMPTS is None:
        examples = load_prompts(REFINE_PROMPTS_PATH)
        RANDOM_PROMPTS = RANDOM_PROMPTS_TEMPLATE.format("\n".join(examples))

    response = get_openai_response(RANDOM_PROMPTS, "Generate one example.")
    return response


def refine_prompt_by_openai(prompt):
    global REFINE_PROMPTS
    if REFINE_PROMPTS is None:
        examples = load_prompts(REFINE_PROMPTS_PATH)
        REFINE_PROMPTS = REFINE_PROMPTS_TEMPLATE.format("\n".join(examples))

    response = get_openai_response(REFINE_PROMPTS, prompt)
    return response


def has_openai_key():
    return "OPENAI_API_KEY" in os.environ


def refine_prompts_by_openai(prompts):
    new_prompts = []
    for prompt in prompts:
        try:
            if prompt.strip() == "":
                new_prompt = get_random_prompt_by_openai()
                print(f"[Info] Empty prompt detected, generate random prompt: {new_prompt}")
            else:
                new_prompt = refine_prompt_by_openai(prompt)
                print(f"[Info] Refine prompt: {prompt} -> {new_prompt}")
            new_prompts.append(new_prompt)
        except Exception as e:
            print(f"[Warning] Failed to refine prompt: {prompt} due to {e}")
            new_prompts.append(prompt)
    return new_prompts


FIRST_FRAME_PROMPT_TEMPLATE_WITH_INFO = """
Given the first frame of the video, describe this video and its style in a very detailed manner. Some information about the video is:
'{}'.

Describe the video and its style in a very detailed manner. Pay attention to all objects in the video. You must describe what the human character is doing with action in the video, for instance, talk, walk, blink, laugh, sing or anything else that involves movements in the video. Your description must make it easy for this vide to have human movements, instead of being motionless. The description should be useful for AI to re-generate the video. The description should be no more than six sentences.

"""

FIRST_FRAME_PROMPT_TEMPLATE = """
Given the first frame of the video, you need to generate one input prompt for video generation task. The prompt should be suitable for generating a video using the given image as the first frame.

Describe the video and its style in a very detailed manner. Pay attention to all objects in the video. You must describe what the human character is doing with action in the video, for instance, talk, walk, blink, laugh, sing or anything else that involves movements in the video. Your description must make it easy for this vide to have human movements, instead of being motionless. The description should be useful for AI to re-generate the video. The description should be no more than six sentences.
"""


def to_base64(image_tensor):
    buffer = BytesIO()
    pil_image = transforms.ToPILImage()(image_tensor)
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


LLAVA_PREFIX = [
    "The video shows ",
    "The video captures ",
    "The video features ",
    "The video depicts ",
    "The video presents ",
    "The video features ",
    "The video is ",
    "In the video, ",
    "The image shows ",
    "The image captures ",
    "The image features ",
    "The image depicts ",
    "The image presents ",
    "The image features ",
    "The image is ",
    "The image portrays ",
    "In the image, ",
]


def remove_caption_prefix(caption):
    for prefix in LLAVA_PREFIX:
        if caption.startswith(prefix) or caption.startswith(prefix.lower()):
            caption = caption[len(prefix) :].strip()
            if caption[0].islower():
                caption = caption[0].upper() + caption[1:]
            return caption
    return caption


def get_caption(frame, prompt):
    global OPENAI_CLIENT
    if OPENAI_CLIENT is None:
        from openai import OpenAI

        OPENAI_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}", "detail": "low"}},
                ],
            },
        ],
        max_tokens=300,
        top_p=0.1,
    )
    caption = response.choices[0].message.content
    caption = caption.replace("\n", " ")
    caption = remove_caption_prefix(caption).replace(" image ", " video ")
    return caption


def refine_batched_prompts_with_images(prompts, images):
    new_prompts = []
    for prompt, image in zip(prompts, images):
        try:
            if prompt.strip() == "":
                new_prompt = get_random_prompt_with_image(image)
                print(f"[Info] Empty prompt detected, generate random prompt: {new_prompt}")
            else:
                new_prompt = refine_prompt_with_image(prompt, image)
                print(f"[Info] Refine prompt: {prompt} -> {new_prompt}")
            new_prompts.append(new_prompt)
        except Exception as e:
            print(f"[Warning] Failed to refine prompt: {prompt} due to {e}")
            new_prompts.append(prompt)
    return new_prompts


def refine_prompt_with_image(prompt, image):
    # check api keys
    if has_openai_key():
        os.environ.get("OPENAI_API_KEY")
    else:
        print("no openai api key found, prompt not refined")
        return prompt
    frame = to_base64(image)
    caption = get_caption(frame, FIRST_FRAME_PROMPT_TEMPLATE_WITH_INFO.format(prompt))
    return caption


def get_random_prompt_with_image(image):
    # check api keys
    if has_openai_key():
        os.environ.get("OPENAI_API_KEY")
    else:
        print("no openai api key found, prompt not refined")
        return prompt

    frame = to_base64(image)
    caption = get_caption(frame, FIRST_FRAME_PROMPT_TEMPLATE)
    return caption


def add_watermark(
    input_video_path, watermark_image_path="./assets/images/watermark/watermark.png", output_video_path=None
):
    # execute this command in terminal with subprocess
    # return if the process is successful
    if output_video_path is None:
        output_video_path = input_video_path.replace(".mp4", "_watermark.mp4")
    cmd = f'ffmpeg -y -i {input_video_path} -i {watermark_image_path} -filter_complex "[1][0]scale2ref=oh*mdar:ih*0.1[logo][video];[video][logo]overlay" {output_video_path}'
    exit_code = os.system(cmd)
    is_success = exit_code == 0
    return is_success


def super_resolution(input_video_path, sr=2):
    temp_dir = tempfile.TemporaryDirectory()
    cmd = f"python -m tools.repair.inference_realesrgan_video -n RealESRGAN_x4plus -s 2 -i {input_video_path} -o {temp_dir.name}"
    os.system(cmd)
    output_video_path = os.path.join(temp_dir.name, os.path.basename(input_video_path).split(".")[0] + "_out.mp4")
    dst_video_path = os.path.join(
        os.path.dirname(input_video_path), os.path.basename(input_video_path).split(".")[0] + f"_sr{sr:.0f}.mp4"
    )
    shutil.copyfile(output_video_path, dst_video_path)
    temp_dir.cleanup()
    return dst_video_path


def deflicker_video_local_brightness(input_video_path, output_video_path, smoothing_window=30, block_size=32):
    # 打开输入视频
    cap = cv2.VideoCapture(input_video_path)

    # 获取视频的基本信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 设置输出视频的编解码器和格式
    fourcc = cv2.VideoWriter_fourcc(*"h264")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 初始化存储每帧局部亮度信息的数组
    local_brightness_list = []

    # 读取视频帧并计算局部亮度
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 将图像划分为小块，并计算每块的平均亮度
        local_brightness = []
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                block = gray[y : y + block_size, x : x + block_size]
                mean_brightness = np.mean(block)
                local_brightness.append(mean_brightness)

        local_brightness_list.append(local_brightness)

    # 将局部亮度进行平滑处理
    local_brightness_array = np.array(local_brightness_list)

    # 在前后添加填充
    assert total_frames % 2 == 1, "The number of frames should be odd."
    pad_width = smoothing_window // 2
    padded_local_brightness = np.pad(local_brightness_array, ((pad_width, pad_width - 1), (0, 0)), mode="edge")

    # 创建一个存储平滑亮度的数组，大小与local_brightness_array相同
    smoothed_local_brightness = np.zeros_like(local_brightness_array)

    # 对每个局部块的亮度进行平滑处理
    for i in range(local_brightness_array.shape[1]):
        # 使用卷积进行平滑处理
        smoothed_brightness = np.convolve(
            padded_local_brightness[:, i], np.ones(smoothing_window) / smoothing_window, mode="valid"
        )
        smoothed_local_brightness[:, i] = smoothed_brightness

    # 对每一帧应用局部亮度校正
    for i in range(len(frames)):
        frame = frames[i].copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 逐块调整亮度
        idx = 0
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                block = gray[y : y + block_size, x : x + block_size]
                current_brightness = np.mean(block)
                brightness_ratio = smoothed_local_brightness[i, idx] / current_brightness
                frame[y : y + block_size, x : x + block_size] = np.clip(
                    frame[y : y + block_size, x : x + block_size] * brightness_ratio, 0, 255
                ).astype(np.uint8)
                idx += 1

        out.write(frame)

    # 释放资源
    cap.release()
    out.release()
    print(f"Deflickered video saved as {output_video_path}")


def deflicker(input_video_path):
    deflicker_video_local_brightness(input_video_path, input_video_path.replace(".mp4", "_deflicker.mp4"))
    return input_video_path


GIGABYTE = 1024**3


def print_memory_usage(prefix: str, device: torch.device):
    torch.cuda.synchronize()
    max_memory_allocated = torch.cuda.max_memory_allocated(device)
    max_memory_reserved = torch.cuda.max_memory_reserved(device)
    print(f"{prefix}: max memory allocated: {max_memory_allocated / GIGABYTE:.4f} GB")
    print(f"{prefix}: max memory reserved: {max_memory_reserved / GIGABYTE:.4f} GB")


def easy_data(csv):
    from opensora.registry import DATASETS, build_module

    dataset = build_module(
        {
            "type": "VariableVideoTextDataset",
            "transform_name": "resize_crop",
            "data_path": csv,
        },
        DATASETS,
    )

    return dataset["0-113-360-640"]


def prep_ref_and_mask(cond_type, condition_frame_length, refs, target_shape, loop, device, dtype):
    """
    prepare the mask_index and reference for the 1st loop
    Input:
        loop: total number of loops to do
    """
    latent_t = target_shape[2]

    if cond_type is None:
        mask_index = []

    elif cond_type == "v2v_head":
        min_ref_length = min([ref[0].shape[1] for ref in refs])
        condition_frame_length = min(
            min(condition_frame_length, min_ref_length), latent_t
        )  # ensure condition frame is no more than generated length
        mask_index = [i for i in range(condition_frame_length)]
    elif cond_type == "i2v_head" or cond_type == "i2v_loop":
        mask_index = [0]  # update mask on last frame lfor i2v_loop
    elif cond_type == "i2v_tail":
        if loop == 1:
            mask_index = [-1]  # update mask to be positive later
        else:
            mask_index = []  # cond on last frame in final loop
    else:
        raise NotImplementedError

    # prep ref in correct shape
    ref = torch.zeros(target_shape, device=device, dtype=dtype)

    if len(mask_index) > 0:
        b = target_shape[0]
        for b_i in range(b):
            ref[b_i, :, mask_index] = refs[b_i][0][:, mask_index].unsqueeze(0)

    # get finalized mask_index, except i2v_tail and i2v_loop intermediate loops will update later
    if cond_type == "i2v_loop" and loop <= 1:
        b = target_shape[0]
        for b_i in range(b):
            if len(refs[0]) == 1:  # if only 1 ref, use last frame
                ref[b_i, :, -1] = refs[b_i][0][:, -1].unsqueeze(0)
            else:
                ref[b_i, :, -1] = refs[b_i][1][:, 0].unsqueeze(0)  # CHANGED TO USE IMAGE
        mask_index.append(latent_t - 1)
    if cond_type == "i2v_tail" and loop <= 1:
        mask_index = [latent_t - 1]

    return ref, mask_index


def prep_ref_and_update_mask_in_loop(
    cond_type, condition_frame_length, samples, refs, target_shape, is_last_loop, device, dtype
):
    latent_t = target_shape[2]
    # cond frames from last generation
    loop_cond_index = [i for i in range(-condition_frame_length, 0)]

    # get ref in correct shape
    ref = torch.zeros(target_shape, device=device, dtype=dtype)
    ref_cut = samples[:, :, loop_cond_index].to(device=device, dtype=dtype)
    mask_index = [i for i in range(condition_frame_length)]
    ref[:, :, mask_index] = ref_cut

    if cond_type == "i2v_loop" or cond_type == "i2v_tail" and is_last_loop:
        b = target_shape[0]
        for b_i in range(b):
            if len(refs[b_i]) == 1:  # if only 1 reference passed, use last frame
                ref[b_i, :, -1] = refs[b_i][0][:, -1].unsqueeze(0).to(device=device, dtype=dtype)
            else:  # use the last frame (either video or image) of second reference
                ref[b_i, :, -1] = refs[b_i][1][:, 0].unsqueeze(0).to(device=device, dtype=dtype)

        mask_index.append(latent_t - 1)  # mask_index for final loop

    return ref, mask_index

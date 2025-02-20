# code modified based on https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/playground/demo/video_demo.py

import argparse
import base64
import csv
import math
import os
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from decord import VideoReader, cpu
from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, BitsAndBytesConfig

warnings.filterwarnings("ignore")

PAD_TOKEN_ID = 151643


# Function to initialize the distributed environment
def setup(rank, world_size):
    print(f"Setting up process {rank} of {world_size}")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    # Initialize the process group for communication
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


# Cleanup after inference is done
def cleanup():
    dist.destroy_process_group()


class VideoDataset:
    def __init__(self, df, args, rank, world_size, image_processor, model, tokenizer):
        self.df = df
        self.rank = rank
        self.world_size = world_size
        self.args = args
        self.image_processor = image_processor
        self.model = model
        self.tokenizer = tokenizer
        self.length = len([i for i in range(self.rank, len(self.df), self.world_size)])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.df.iloc[self.rank + idx * self.world_size]
        video_path = row["path"]
        info = row
        sample_set = {}
        question = self.args.prompt
        sample_set["video_name"] = video_path

        if os.path.exists(video_path):
            video, frame_time, video_time = load_video(video_path, self.args)
            video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half()
            sample_set["video"] = video
            sample_set["frame_time"] = frame_time
            sample_set["video_time"] = video_time
            sample_set["info"] = info

            if self.args.add_time_instruction:
                time_instruction = (
                    f"The video lasts for {video_time:.2f} seconds, and "
                    f"{self.args.for_get_frames_num} frames are uniformly sampled from it. "
                    f"These frames are located at {frame_time}. "
                    f"Please answer the following questions related to this video."
                )
                qs = f"{time_instruction}\n{question}"
            else:
                qs = question
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[self.args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
                0
            )
            # print("input_ids", input_ids)
            sample_set["input_ids"] = input_ids
            attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long()
            sample_set["attention_masks"] = attention_masks

        return sample_set


def collate_fn(batch):
    # Collate function to handle dynamic padding or combining elements of the batch
    videos = [item["video"] for item in batch if "video" in item]
    input_ids = [item["input_ids"] for item in batch]
    max_len = max([item.shape[1] for item in input_ids])
    # pad token id PAD_TOKEN_ID
    input_ids = [torch.nn.functional.pad(item, (max_len - item.shape[1], 0), value=PAD_TOKEN_ID) for item in input_ids]
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = [item["attention_masks"] for item in batch]
    attention_masks = [torch.nn.functional.pad(item, (max_len - item.shape[1], 0), value=0) for item in attention_masks]
    attention_masks = torch.cat(attention_masks, dim=0)

    video_names = [item["video_name"] for item in batch]
    frame_times = [item["frame_time"] for item in batch]
    video_times = [item["video_time"] for item in batch]
    infos = [item["info"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "videos": videos,
        "video_names": video_names,
        "frame_times": frame_times,
        "video_times": video_times,
        "infos": infos,
    }


def create_dataloader(df, args, rank, world_size, image_processor, model, tokenizer):
    dataset = VideoDataset(df, args, rank, world_size, image_processor, model, tokenizer)
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers
    )


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--data_file", help="Path to the video dataset file.", required=True)
    parser.add_argument("--output_folder", help="Path to the output file.", required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument(
        "--image_grid_pinpoints",
        type=str,
        default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]",
    )
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == "true"), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--load_8bit", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--add_time_instruction", type=str, default=False)
    return parser.parse_args()


def load_video(video_path, args):
    if args.for_get_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > args.for_get_frames_num or args.force_sample:
        sample_fps = args.for_get_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()

    return spare_frames, frame_time, video_time


def load_video_base64(path):
    video = cv2.VideoCapture(path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    # print(len(base64Frames), "frames read.")
    return base64Frames


def run_inference(rank, world_size, args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    # Set model configuration parameters if they exist
    if args.overwrite == True:
        overwrite_config = {}
        overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
        overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
        overwrite_config["mm_newline_position"] = args.mm_newline_position

        cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

        # import pdb;pdb.set_trace()
        if "qwen" not in args.model_path.lower():
            if "224" in cfg_pretrained.mm_vision_tower:
                # suppose the length of text tokens is around 1000, from bo's report
                least_token_number = args.for_get_frames_num * (16 // args.mm_spatial_pool_stride) ** 2 + 1000
            else:
                least_token_number = args.for_get_frames_num * (24 // args.mm_spatial_pool_stride) ** 2 + 1000

            scaling_factor = math.ceil(least_token_number / 4096)
            if scaling_factor >= 2:
                if "vicuna" in cfg_pretrained._name_or_path.lower():
                    print(float(scaling_factor))
                    overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor
        if args.load_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                args.model_path,
                args.model_base,
                model_name,
                device_map=device,
                quantization_config=quantization_config,
                overwrite_config=overwrite_config,
            )
        else:
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                args.model_path, args.model_base, model_name, device_map=device, overwrite_config=overwrite_config
            )
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, args.model_base, model_name, device_map=device
        )

    if tokenizer.pad_token_id is None:
        if "qwen" in tokenizer.name_or_path.lower():
            # print("Setting pad token to bos token for qwen model.")
            tokenizer.pad_token_id = PAD_TOKEN_ID

    if args.batch_size > 1:
        tokenizer.padding_side = "left"
        model.config.tokenizer_padding_side = "left"

    # model = DDP(model, device_ids=[rank])

    # import pdb;pdb.set_trace()
    if getattr(model.config, "force_sample", None) is not None:
        args.force_sample = model.config.force_sample
    else:
        args.force_sample = False

    if getattr(model.config, "add_time_instruction", None) is not None:
        args.add_time_instruction = model.config.add_time_instruction
    else:
        args.add_time_instruction = False

    df = pd.read_csv(args.data_file)
    data_name = os.path.basename(args.data_file).split(".csv")[0]
    column_names = df.columns.to_list()
    if "text" not in column_names:
        column_names.append("text")
    text_column_index = column_names.index("text")

    output_file = os.path.join(args.output_folder, f"{data_name}_{rank}.csv")
    with open(output_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(column_names)

        dataloader = create_dataloader(df, args, rank, world_size, image_processor, model, tokenizer)

        for batch in tqdm(dataloader):
            videos = [item.to(device) for item in batch["videos"]]
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)
            infos = batch["infos"]
            stop_str = "###"
            with torch.inference_mode():
                modalities = ["video"] * len(videos)
                if "mistral" not in cfg_pretrained._name_or_path.lower():
                    output_ids = model.generate(
                        inputs=input_ids,
                        images=videos,
                        attention_mask=attention_masks,
                        modalities=modalities,
                        do_sample=False,
                        temperature=0.0,
                        max_new_tokens=1024,
                        top_p=0.1,
                        num_beams=1,
                        use_cache=True,
                    )
                    # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
                else:
                    output_ids = model.generate(
                        inputs=input_ids,
                        images=videos,
                        attention_mask=attention_masks,
                        modalities=modalities,
                        do_sample=False,
                        temperature=0.0,
                        max_new_tokens=1024,
                        top_p=0.1,
                        num_beams=1,
                        use_cache=True,
                    )
                    # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True)
            # print("output_ids", output_ids)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            # print("outputs", outputs)
            outputs = [output.split(stop_str)[0].strip() for output in outputs]
            if len(infos[0]) < len(column_names):
                for i in range(len(infos)):
                    infos[i].append(outputs[i])
            else:
                for i in range(len(infos)):
                    infos[i][text_column_index] = outputs[i]

            # write to csv
            for row in infos:
                csvwriter.writerow(row)

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running inference on {world_size} GPUs.")
    args = parse_args()

    # Spawn one process per GPU
    mp.spawn(run_inference, args=(world_size, args), nprocs=world_size, join=True)

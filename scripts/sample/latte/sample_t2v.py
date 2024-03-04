# All rights reserved.
# Copyright 2024 Vchitect/Latte
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# modified from https://github.com/Vchitect/Latte/blob/main/sample/sample_t2v.py

import argparse
import os
import sys

import torch
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from transformers import T5EncoderModel, T5Tokenizer

sys.path.append(os.path.split(sys.path[0])[0])
import imageio
from pipeline_videogen import VideoGenPipeline
from utils import save_video_grid

from download import find_model
from open_sora.modeling import LatteT2V


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="The path to the pretrained model files")
    parser.add_argument("--checkpoint", type=str, required=True, help="The path to the t2v.pt file.")
    parser.add_argument("--output_path", type=str, required=True, help="The path to save the output")

    # generation configs
    parser.add_argument(
        "--text_prompt", type=str, nargs="+", required=True, help="The text prompt to generate the video."
    )
    parser.add_argument("--video_length", type=int, default=16, help="The number of frames in the video.")
    parser.add_argument("--image_height", type=int, default=256, help="The size of the generated images.")
    parser.add_argument("--image_width", type=int, default=256, help="The size of the generated images.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="The scale of the guidance loss.")
    parser.add_argument("--sample_method", type=str, default="PNDM", help="The sampling method to use.")
    parser.add_argument("--num_sampling_steps", type=int, default=50, help="The number of sampling steps.")
    parser.add_argument(
        "--enable_temporal_attentions", action="store_true", default=True, help="Whether to enable temporal attentions."
    )
    parser.add_argument(
        "--enable_vae_temporal_decoder",
        action="store_true",
        default=True,
        help="Whether to enable the VAE temporal decoder.",
    )

    # Scheduler configs
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--variance_type", type=str, default="learned_range")

    args = parser.parse_args()
    return args


def main(args):
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transformer_model = LatteT2V.from_pretrained_2d(
        args.model_path, subfolder="transformer", video_length=args.video_length
    ).to(device, dtype=torch.float16)
    state_dict = find_model(args.checkpoint)
    transformer_model.load_state_dict(state_dict["model"])

    if args.enable_vae_temporal_decoder:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            args.model_path, subfolder="vae_temporal_decoder", torch_dtype=torch.float16
        ).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float16).to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        args.model_path, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)

    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    if args.sample_method == "DDIM":
        scheduler = DDIMScheduler.from_pretrained(
            args.model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "EulerDiscrete":
        scheduler = EulerDiscreteScheduler.from_pretrained(
            args.model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "DDPM":
        scheduler = DDPMScheduler.from_pretrained(
            args.model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "DPMSolverMultistep":
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            args.model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "DPMSolverSinglestep":
        scheduler = DPMSolverSinglestepScheduler.from_pretrained(
            args.model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "PNDM":
        scheduler = PNDMScheduler.from_pretrained(
            args.model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "HeunDiscrete":
        scheduler = HeunDiscreteScheduler.from_pretrained(
            args.model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "EulerAncestralDiscrete":
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            args.model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "DEISMultistep":
        scheduler = DEISMultistepScheduler.from_pretrained(
            args.model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "KDPM2AncestralDiscrete":
        scheduler = KDPM2AncestralDiscreteScheduler.from_pretrained(
            args.model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )

    videogen_pipeline = VideoGenPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler, transformer=transformer_model
    ).to(device)
    # videogen_pipeline.enable_xformers_memory_efficient_attention()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    video_grids = []
    for prompt in args.text_prompt:
        print("Processing the ({}) prompt".format(prompt))
        videos = videogen_pipeline(
            prompt,
            video_length=args.video_length,
            height=args.image_height,
            width=args.image_width,
            num_inference_steps=args.num_sampling_steps,
            guidance_scale=args.guidance_scale,
            enable_temporal_attentions=args.enable_temporal_attentions,
            num_images_per_prompt=1,
            mask_feature=True,
            enable_vae_temporal_decoder=args.enable_vae_temporal_decoder,
        ).video
        try:
            save_path = os.path.join(args.output_path, prompt.replace(" ", "_") + "_webv-imageio.mp4")
            imageio.mimwrite(save_path, videos[0], fps=8, quality=9)  # highest quality is 10, lowest is 0
        except:
            print("Error when saving {}".format(prompt))
        video_grids.append(videos)
    video_grids = torch.cat(video_grids, dim=0)

    video_grids = save_video_grid(video_grids)

    # torchvision.io.write_video(args.output_path + '_%04d' % args.run_time + '-.mp4', video_grids, fps=6)
    save_path = os.path.join(args.output_path, "grid.mp4")
    imageio.mimwrite(save_path, video_grids, fps=8, quality=5)
    print("save path {}".format(abspath(args.output_path)))

    # save_videos_grid(video, f"./{prompt}.gif")


if __name__ == "__main__":
    args = parse_args()
    main(args)

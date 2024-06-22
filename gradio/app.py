#!/usr/bin/env python
"""
This script runs a Gradio App for the Open-Sora model.

Usage:
    python demo.py <config-path>
"""

import argparse
import datetime
import importlib
import os
import subprocess
import sys
from tempfile import NamedTemporaryFile

import spaces
import torch

import gradio as gr

MODEL_TYPES = ["v1.2-stage3"]
WATERMARK_PATH = "./assets/images/watermark/watermark.png"
CONFIG_MAP = {
    "v1.2-stage3": "configs/opensora-v1-2/inference/sample.py",
}
HF_STDIT_MAP = {"v1.2-stage3": "hpcai-tech/OpenSora-STDiT-v3"}


# ============================
# Prepare Runtime Environment
# ============================
def install_dependencies(enable_optimization=False):
    """
    Install the required dependencies for the demo if they are not already installed.
    """

    def _is_package_available(name) -> bool:
        try:
            importlib.import_module(name)
            return True
        except (ImportError, ModuleNotFoundError):
            return False

    if enable_optimization:
        # install flash attention
        if not _is_package_available("flash_attn"):
            subprocess.run(
                f"{sys.executable} -m pip install flash-attn --no-build-isolation",
                env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
                shell=True,
            )

        # install apex for fused layernorm
        if not _is_package_available("apex"):
            subprocess.run(
                f'{sys.executable} -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git',
                shell=True,
            )

        # install ninja
        if not _is_package_available("ninja"):
            subprocess.run(f"{sys.executable} -m pip install ninja", shell=True)

        # install xformers
        if not _is_package_available("xformers"):
            subprocess.run(
                f"{sys.executable} -m pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers",
                shell=True,
            )


# ============================
# Model-related
# ============================
def read_config(config_path):
    """
    Read the configuration file.
    """
    from mmengine.config import Config

    return Config.fromfile(config_path)


def build_models(model_type, config, enable_optimization=False):
    """
    Build the models for the given model type and configuration.
    """
    # build vae
    from opensora.registry import MODELS, build_module

    vae = build_module(config.vae, MODELS).cuda()

    # build text encoder
    text_encoder = build_module(config.text_encoder, MODELS)  # T5 must be fp32
    text_encoder.t5.model = text_encoder.t5.model.cuda()

    # build stdit
    # we load model from HuggingFace directly so that we don't need to
    # handle model download logic in HuggingFace Space
    from opensora.models.stdit.stdit3 import STDiT3

    model_kwargs = {k: v for k, v in config.model.items() if k not in ("type", "from_pretrained", "force_huggingface")}
    stdit = STDiT3.from_pretrained(HF_STDIT_MAP[model_type], **model_kwargs)
    stdit = stdit.cuda()

    # build scheduler
    from opensora.registry import SCHEDULERS

    scheduler = build_module(config.scheduler, SCHEDULERS)

    # hack for classifier-free guidance
    text_encoder.y_embedder = stdit.y_embedder

    # move modelst to device
    vae = vae.to(torch.bfloat16).eval()
    text_encoder.t5.model = text_encoder.t5.model.eval()  # t5 must be in fp32
    stdit = stdit.to(torch.bfloat16).eval()

    # clear cuda
    torch.cuda.empty_cache()
    return vae, text_encoder, stdit, scheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        default="v1.2-stage3",
        choices=MODEL_TYPES,
        help=f"The type of model to run for the Gradio App, can only be {MODEL_TYPES}",
    )
    parser.add_argument("--output", default="./outputs", type=str, help="The path to the output folder")
    parser.add_argument("--port", default=None, type=int, help="The port to run the Gradio App on.")
    parser.add_argument("--host", default="0.0.0.0", type=str, help="The host to run the Gradio App on.")
    parser.add_argument("--share", action="store_true", help="Whether to share this gradio demo.")
    parser.add_argument(
        "--enable-optimization",
        action="store_true",
        help="Whether to enable optimization such as flash attention and fused layernorm",
    )
    return parser.parse_args()


# ============================
# Main Gradio Script
# ============================
# as `run_inference` needs to be wrapped by `spaces.GPU` and the input can only be the prompt text
# so we can't pass the models to `run_inference` as arguments.
# instead, we need to define them globally so that we can access these models inside `run_inference`

# read config
args = parse_args()
config = read_config(CONFIG_MAP[args.model_type])
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# make outputs dir
os.makedirs(args.output, exist_ok=True)

# disable torch jit as it can cause failure in gradio SDK
# gradio sdk uses torch with cuda 11.3
torch.jit._state.disable()

# set up
install_dependencies(enable_optimization=args.enable_optimization)

# import after installation
from opensora.datasets import IMG_FPS, save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_random_prompt_by_openai,
    has_openai_key,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from opensora.utils.misc import to_torch_dtype

# some global variables
dtype = to_torch_dtype(config.dtype)
device = torch.device("cuda")

# build model
vae, text_encoder, stdit, scheduler = build_models(
    args.model_type, config, enable_optimization=args.enable_optimization
)


def run_inference(
    mode,
    prompt_text,
    resolution,
    aspect_ratio,
    length,
    motion_strength,
    aesthetic_score,
    use_motion_strength,
    use_aesthetic_score,
    camera_motion,
    reference_image,
    refine_prompt,
    fps,
    num_loop,
    seed,
    sampling_steps,
    cfg_scale,
):
    if prompt_text is None or prompt_text == "":
        gr.Warning("Your prompt is empty, please enter a valid prompt")
        return None

    torch.manual_seed(seed)
    with torch.inference_mode():
        # ======================
        # 1. Preparation arguments
        # ======================
        # parse the inputs
        # frame_interval must be 1 so  we ignore it here
        image_size = get_image_size(resolution, aspect_ratio)

        # compute generation parameters
        if mode == "Text2Image":
            num_frames = 1
            fps = IMG_FPS
        else:
            num_frames = config.num_frames
            num_frames = get_num_frames(length)

        condition_frame_length = int(num_frames / 17 * 5 / 3)
        condition_frame_edit = 0.0

        input_size = (num_frames, *image_size)
        latent_size = vae.get_latent_size(input_size)
        multi_resolution = "OpenSora"
        align = 5

        # == prepare mask strategy ==
        if mode == "Text2Image":
            mask_strategy = [None]
        elif mode == "Text2Video":
            if reference_image is not None:
                mask_strategy = ["0"]
            else:
                mask_strategy = [None]
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # == prepare reference ==
        if mode == "Text2Image":
            refs = [""]
        elif mode == "Text2Video":
            if reference_image is not None:
                # save image to disk
                from PIL import Image

                im = Image.fromarray(reference_image)
                temp_file = NamedTemporaryFile(suffix=".png")
                im.save(temp_file.name)
                refs = [temp_file.name]
            else:
                refs = [""]
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # == get json from prompts ==
        batch_prompts = [prompt_text]
        batch_prompts, refs, mask_strategy = extract_json_from_prompts(batch_prompts, refs, mask_strategy)

        # == get reference for condition ==
        refs = collect_references_batch(refs, vae, image_size)

        # == multi-resolution info ==
        model_args = prepare_multi_resolution_info(
            multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
        )

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
        if refine_prompt:
            # check if openai key is provided
            if not has_openai_key():
                gr.Warning("OpenAI API key is not provided, the prompt will not be enhanced.")
            else:
                for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                    batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)

        # process scores
        aesthetic_score = aesthetic_score if use_aesthetic_score else None
        motion_strength = motion_strength if use_motion_strength and mode != "Text2Image" else None
        camera_motion = None if camera_motion == "none" or mode == "Text2Image" else camera_motion
        # 2. append score
        for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
            batched_prompt_segment_list[idx] = append_score_to_prompts(
                prompt_segment_list,
                aes=aesthetic_score,
                flow=motion_strength,
                camera_motion=camera_motion,
            )

        # 3. clean prompt with T5
        for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
            batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

        # 4. merge to obtain the final prompt
        batch_prompts = []
        for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
            batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

        # =========================
        # Generate image/video
        # =========================
        video_clips = []

        for loop_i in range(num_loop):
            # 4.4 sample in hidden space
            batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

            # == loop ==
            if loop_i > 0:
                refs, mask_strategy = append_generated(
                    vae, video_clips[-1], refs, mask_strategy, loop_i, condition_frame_length, condition_frame_edit
                )

            # == sampling ==
            z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
            masks = apply_mask_strategy(z, refs, mask_strategy, loop_i, align=align)

            # 4.6. diffusion sampling
            # hack to update num_sampling_steps and cfg_scale
            scheduler_kwargs = config.scheduler.copy()
            scheduler_kwargs.pop("type")
            scheduler_kwargs["num_sampling_steps"] = sampling_steps
            scheduler_kwargs["cfg_scale"] = cfg_scale

            scheduler.__init__(**scheduler_kwargs)
            samples = scheduler.sample(
                stdit,
                text_encoder,
                z=z,
                prompts=batch_prompts_loop,
                device=device,
                additional_args=model_args,
                progress=True,
                mask=masks,
            )
            samples = vae.decode(samples.to(dtype), num_frames=num_frames)
            video_clips.append(samples)

        # =========================
        # Save output
        # =========================
        video_clips = [val[0] for val in video_clips]
        for i in range(1, num_loop):
            video_clips[i] = video_clips[i][:, dframe_to_frame(condition_frame_length) :]
        video = torch.cat(video_clips, dim=1)
        current_datetime = datetime.datetime.now()
        timestamp = current_datetime.timestamp()
        save_path = os.path.join(args.output, f"output_{timestamp}")
        saved_path = save_sample(video, save_path=save_path, fps=24)
        torch.cuda.empty_cache()

        # add watermark
        # all watermarked videos should have a _watermarked suffix
        if mode != "Text2Image" and os.path.exists(WATERMARK_PATH):
            watermarked_path = saved_path.replace(".mp4", "_watermarked.mp4")
            success = add_watermark(saved_path, WATERMARK_PATH, watermarked_path)
            if success:
                return watermarked_path
            else:
                return saved_path
        else:
            return saved_path


@spaces.GPU(duration=200)
def run_image_inference(
    prompt_text,
    resolution,
    aspect_ratio,
    length,
    motion_strength,
    aesthetic_score,
    use_motion_strength,
    use_aesthetic_score,
    camera_motion,
    reference_image,
    refine_prompt,
    fps,
    num_loop,
    seed,
    sampling_steps,
    cfg_scale,
):
    return run_inference(
        "Text2Image",
        prompt_text,
        resolution,
        aspect_ratio,
        length,
        motion_strength,
        aesthetic_score,
        use_motion_strength,
        use_aesthetic_score,
        camera_motion,
        reference_image,
        refine_prompt,
        fps,
        num_loop,
        seed,
        sampling_steps,
        cfg_scale,
    )


@spaces.GPU(duration=200)
def run_video_inference(
    prompt_text,
    resolution,
    aspect_ratio,
    length,
    motion_strength,
    aesthetic_score,
    use_motion_strength,
    use_aesthetic_score,
    camera_motion,
    reference_image,
    refine_prompt,
    fps,
    num_loop,
    seed,
    sampling_steps,
    cfg_scale,
):
    # if (resolution == "480p" and length == "16s") or \
    #     (resolution == "720p" and length in ["8s", "16s"]):
    #     gr.Warning("Generation is interrupted as the combination of 480p and 16s will lead to CUDA out of memory")
    # else:
    return run_inference(
        "Text2Video",
        prompt_text,
        resolution,
        aspect_ratio,
        length,
        motion_strength,
        aesthetic_score,
        use_motion_strength,
        use_aesthetic_score,
        camera_motion,
        reference_image,
        refine_prompt,
        fps,
        num_loop,
        seed,
        sampling_steps,
        cfg_scale,
    )


def generate_random_prompt():
    if "OPENAI_API_KEY" not in os.environ:
        gr.Warning("Your prompt is empty and the OpenAI API key is not provided, please enter a valid prompt")
        return None
    else:
        prompt_text = get_random_prompt_by_openai()
        return prompt_text


def main():
    # create demo
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr.HTML(
                    """
                <div style='text-align: center;'>
                    <p align="center">
                        <img src="https://github.com/hpcaitech/Open-Sora/raw/main/assets/readme/icon.png" width="250"/>
                    </p>
                    <div style="display: flex; gap: 10px; justify-content: center;">
                        <a href="https://github.com/hpcaitech/Open-Sora/stargazers"><img src="https://img.shields.io/github/stars/hpcaitech/Open-Sora?style=social"></a>
                        <a href="https://hpcaitech.github.io/Open-Sora/"><img src="https://img.shields.io/badge/Gallery-View-orange?logo=&amp"></a>
                        <a href="https://discord.gg/kZakZzrSUT"><img src="https://img.shields.io/badge/Discord-join-blueviolet?logo=discord&amp"></a>
                        <a href="https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-247ipg9fk-KRRYmUl~u2ll2637WRURVA"><img src="https://img.shields.io/badge/Slack-ColossalAI-blueviolet?logo=slack&amp"></a>
                        <a href="https://twitter.com/yangyou1991/status/1769411544083996787?s=61&t=jT0Dsx2d-MS5vS9rNM5e5g"><img src="https://img.shields.io/badge/Twitter-Discuss-blue?logo=twitter&amp"></a>
                        <a href="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png"><img src="https://img.shields.io/badge/微信-小助手加群-green?logo=wechat&amp"></a>
                        <a href="https://hpc-ai.com/blog/open-sora-v1.0"><img src="https://img.shields.io/badge/Open_Sora-Blog-blue"></a>
                    </div>
                    <h1 style='margin-top: 5px;'>Open-Sora: Democratizing Efficient Video Production for All</h1>
                </div>
                """
                )

        with gr.Row():
            with gr.Column():
                prompt_text = gr.Textbox(label="Prompt", placeholder="Describe your video here", lines=4)
                refine_prompt = gr.Checkbox(
                    value=has_openai_key(), label="Refine prompt with GPT4o", interactive=has_openai_key()
                )
                random_prompt_btn = gr.Button("Random Prompt By GPT4o", interactive=has_openai_key())

                gr.Markdown("## Basic Settings")
                resolution = gr.Radio(
                    choices=["144p", "240p", "360p", "480p", "720p"],
                    value="480p",
                    label="Resolution",
                )
                aspect_ratio = gr.Radio(
                    choices=["9:16", "16:9", "3:4", "4:3", "1:1"],
                    value="9:16",
                    label="Aspect Ratio (H:W)",
                )
                length = gr.Radio(
                    choices=["2s", "4s", "8s", "16s"],
                    value="2s",
                    label="Video Length",
                    info="only effective for video generation, 8s may fail as Hugging Face ZeroGPU has the limitation of max 200 seconds inference time.",
                )

                with gr.Row():
                    seed = gr.Slider(value=1024, minimum=1, maximum=2048, step=1, label="Seed")

                    sampling_steps = gr.Slider(value=30, minimum=1, maximum=200, step=1, label="Sampling steps")
                    cfg_scale = gr.Slider(value=7.0, minimum=0.0, maximum=10.0, step=0.1, label="CFG Scale")

                with gr.Row():
                    with gr.Column():
                        motion_strength = gr.Slider(
                            value=5,
                            minimum=0,
                            maximum=100,
                            step=1,
                            label="Motion Strength",
                            info="only effective for video generation",
                        )
                        use_motion_strength = gr.Checkbox(value=False, label="Enable")

                    with gr.Column():
                        aesthetic_score = gr.Slider(
                            value=6.5,
                            minimum=4,
                            maximum=7,
                            step=0.1,
                            label="Aesthetic",
                            info="effective for text & video generation",
                        )
                        use_aesthetic_score = gr.Checkbox(value=True, label="Enable")

                camera_motion = gr.Radio(
                    value="none",
                    label="Camera Motion",
                    choices=["none", "pan right", "pan left", "tilt up", "tilt down", "zoom in", "zoom out", "static"],
                    interactive=True,
                )

                gr.Markdown("## Advanced Settings")
                with gr.Row():
                    fps = gr.Slider(
                        value=24,
                        minimum=1,
                        maximum=60,
                        step=1,
                        label="FPS",
                        info="This is the frames per seconds for video generation, keep it to 24 if you are not sure",
                    )
                    num_loop = gr.Slider(
                        value=1,
                        minimum=1,
                        maximum=20,
                        step=1,
                        label="Number of Loops",
                        info="This will change the length of the generated video, keep it to 1 if you are not sure",
                    )

                gr.Markdown("## Reference Image")
                reference_image = gr.Image(label="Image (optional)", show_download_button=True)

            with gr.Column():
                output_video = gr.Video(label="Output Video", height="100%")

        with gr.Row():
            image_gen_button = gr.Button("Generate image")
            video_gen_button = gr.Button("Generate video")

        image_gen_button.click(
            fn=run_image_inference,
            inputs=[
                prompt_text,
                resolution,
                aspect_ratio,
                length,
                motion_strength,
                aesthetic_score,
                use_motion_strength,
                use_aesthetic_score,
                camera_motion,
                reference_image,
                refine_prompt,
                fps,
                num_loop,
                seed,
                sampling_steps,
                cfg_scale,
            ],
            outputs=reference_image,
        )
        video_gen_button.click(
            fn=run_video_inference,
            inputs=[
                prompt_text,
                resolution,
                aspect_ratio,
                length,
                motion_strength,
                aesthetic_score,
                use_motion_strength,
                use_aesthetic_score,
                camera_motion,
                reference_image,
                refine_prompt,
                fps,
                num_loop,
                seed,
                sampling_steps,
                cfg_scale,
            ],
            outputs=output_video,
        )
        random_prompt_btn.click(fn=generate_random_prompt, outputs=prompt_text)

    # launch
    demo.queue(max_size=5, default_concurrency_limit=1)
    demo.launch(server_port=args.port, server_name=args.host, share=args.share, max_threads=1)


if __name__ == "__main__":
    main()

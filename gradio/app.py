#!/usr/bin/env python
"""
This script runs a Gradio App for the Open-Sora model.

Usage:
    python demo.py <config-path>
"""

import argparse
import importlib
import os
import subprocess
import sys

import spaces
import torch

import gradio as gr

MODEL_TYPES = ["v1-16x256x256", "v1-HQ-16x256x256", "v1-HQ-16x512x512"]
CONFIG_MAP = {
    "v1-16x256x256": "configs/opensora/inference/16x256x256.py",
    "v1-HQ-16x256x256": "configs/opensora/inference/16x256x256.py",
    "v1-HQ-16x512x512": "configs/opensora/inference/16x512x512.py",
}
HF_STDIT_MAP = {
    "v1-16x256x256": "hpcai-tech/OpenSora-STDiT-v1-16x256x256",
    "v1-HQ-16x256x256": "hpcai-tech/OpenSora-STDiT-v1-HQ-16x256x256",
    "v1-HQ-16x512x512": "hpcai-tech/OpenSora-STDiT-v1-HQ-16x512x512",
}


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

    # flash attention is needed no matter optimization is enabled or not
    # because Hugging Face transformers detects flash_attn is a dependency in STDiT
    # thus, we need to install it no matter what
    if not _is_package_available("flash_attn"):
        subprocess.run(
            f"{sys.executable} -m pip install flash-attn --no-build-isolation",
            env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
            shell=True,
        )

    if enable_optimization:
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


def read_config(config_path):
    """
    Read the configuration file.
    """
    from mmengine.config import Config

    return Config.fromfile(config_path)


def build_models(model_type, config):
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
    from transformers import AutoModel

    stdit = AutoModel.from_pretrained(
        HF_STDIT_MAP[model_type],
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        trust_remote_code=True,
    ).cuda()

    # build scheduler
    from opensora.registry import SCHEDULERS

    scheduler = build_module(config.scheduler, SCHEDULERS)

    # hack for classifier-free guidance
    text_encoder.y_embedder = stdit.y_embedder

    # move modelst to device
    vae = vae.to(torch.float16).eval()
    text_encoder.t5.model = text_encoder.t5.model.eval()  # t5 must be in fp32
    stdit = stdit.to(torch.float16).eval()
    return vae, text_encoder, stdit, scheduler


def get_latent_size(config, vae):
    input_size = (config.num_frames, *config.image_size)
    latent_size = vae.get_latent_size(input_size)
    return latent_size


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        default="v1-HQ-16x256x256",
        choices=MODEL_TYPES,
        help=f"The type of model to run for the Gradio App, can only be {MODEL_TYPES}",
    )
    parser.add_argument("--output", default="./outputs", type=str, help="The path to the output folder")
    parser.add_argument("--port", default=None, type=int, help="The port to run the Gradio App on.")
    parser.add_argument("--host", default=None, type=str, help="The host to run the Gradio App on.")
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

# make outputs dir
os.makedirs(args.output, exist_ok=True)

# disable torch jit as it can cause failure in gradio SDK
# gradio sdk uses torch with cuda 11.3
torch.jit._state.disable()

# set up
install_dependencies(enable_optimization=args.enable_optimization)

# build model
vae, text_encoder, stdit, scheduler = build_models(args.model_type, config)


@spaces.GPU(duration=200)
def run_inference(prompt_text):
    from opensora.datasets import save_sample

    latent_size = get_latent_size(config, vae)
    samples = scheduler.sample(
        stdit,
        text_encoder,
        z_size=(vae.out_channels, *latent_size),
        prompts=[prompt_text],
        device="cuda",
    )

    samples = vae.decode(samples.to(torch.float16))
    filename = f"{args.output}/sample"
    saved_path = save_sample(samples[0], fps=config.fps, save_path=filename)
    return saved_path


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
                prompt_text = gr.Textbox(show_label=False, placeholder="Describe your video here", lines=4)
                submit_button = gr.Button("Generate video")

            with gr.Column():
                output_video = gr.Video()

        submit_button.click(fn=run_inference, inputs=[prompt_text], outputs=output_video)

        gr.Examples(
            examples=[
                [
                    "The video captures the majestic beauty of a waterfall cascading down a cliff into a serene lake. The waterfall, with its powerful flow, is the central focus of the video. The surrounding landscape is lush and green, with trees and foliage adding to the natural beauty of the scene. The camera angle provides a bird's eye view of the waterfall, allowing viewers to appreciate the full height and grandeur of the waterfall. The video is a stunning representation of nature's power and beauty.",
                ],
            ],
            fn=run_inference,
            inputs=[
                prompt_text,
            ],
            outputs=[output_video],
            cache_examples=True,
        )

    # launch
    demo.launch(server_port=args.port, server_name=args.host, share=args.share)


if __name__ == "__main__":
    main()

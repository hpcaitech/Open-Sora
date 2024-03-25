# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import random
import subprocess
import shutil
import time
from typing import List

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from mmengine.config import Config
from mmengine.runner import set_random_seed

from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import merge_args
from opensora.utils.misc import to_torch_dtype

MAX_SEED = np.iinfo(np.int32).max

MODEL_URL = "https://weights.replicate.delivery/default/open-sora/opensora.tar"
WEIGHTS_FOLDER = "pretrained_models"

def download_weights(url, dest, extract=True):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    args = ["pget"]
    if extract:
        args.append("-x")
    subprocess.check_call(args + [url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)
    
def cog_config():
    # taken from 16x512x512.py
    cfg = Config(dict(
        num_frames = 16,
        fps = 24 // 3,
        image_size = (512, 512),
        dtype = "fp16",
        batch_size = 2,
        seed = 42,
        prompt_path = "./assets/texts/t2v_samples.txt",
        save_dir = "./outputs/samples/",
    ))

    cfg.model = dict(
        type="STDiT-XL/2",
        space_scale=1.0,
        time_scale=1.0,
        enable_flashattn=True,
        enable_layernorm_kernel=True,
        from_pretrained="PRETRAINED_MODEL"
    )
    cfg.vae = dict(
        type="VideoAutoencoderKL",
        from_pretrained="stabilityai/sd-vae-ft-ema",
        micro_batch_size=128,
    )
    cfg.text_encoder = dict(
        type="t5",
        from_pretrained="./pretrained_models/t5_ckpts",
        model_max_length=120,
    )
    cfg.scheduler = dict(
        type="iddpm",
        num_sampling_steps=100,
        cfg_scale=7.0,
    )
    return cfg


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # install open sora from github repo
        subprocess.check_call("pip install -q .".split())
        
        # download model
        if not os.path.exists(WEIGHTS_FOLDER):
            download_weights(MODEL_URL, WEIGHTS_FOLDER, extract=True)
        
        # command line arguments from opensora.utils.config_utils
        extra_args = Config({
            'seed': 42,
            'ckpt_path': "pretrained_models/Open-Sora/OpenSora-v1-HQ-16x512x512.pth",
            'batch-size': None,
            'prompt-path': None,
            'save-dir': None,
            'num-sampling-steps': None,
            'cfg_scale': None,
        })
        
        self.cfg = cog_config()
        self.cfg = merge_args(self.cfg, args=extra_args, training=False)

        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.dtype = to_torch_dtype(self.cfg.dtype)

        input_size = (self.cfg.num_frames, *self.cfg.image_size)
        self.vae = build_module(self.cfg.vae, MODELS)
        self.latent_size = self.vae.get_latent_size(input_size)
        self.text_encoder = build_module(self.cfg.text_encoder, MODELS, device=self.device)  # T5 must be fp32
        self.model = build_module(
            self.cfg.model,
            MODELS,
            input_size=self.latent_size,
            in_channels=self.vae.out_channels,
            caption_channels=self.text_encoder.output_dim,
            model_max_length=self.text_encoder.model_max_length,
            dtype=self.dtype,
            enable_sequence_parallelism=False,
        )
        self.text_encoder.y_embedder = self.model.y_embedder  # hack for classifier-free guidance

        self.vae = self.vae.to(self.device, self.dtype).eval()
        self.model = self.model.to(self.device, self.dtype).eval()
        self.scheduler = build_module(self.cfg.scheduler, SCHEDULERS)

        self.model_args = dict()
        if self.cfg.multi_resolution:
            image_size = self.cfg.image_size
            hw = torch.tensor([image_size], device=self.device, dtype=self.dtype).repeat(self.cfg.batch_size, 1)
            ar = torch.tensor([[image_size[0] / image_size[1]]], device=self.device, dtype=self.dtype).repeat(self.cfg.batch_size, 1)
            self.model_args["data_info"] = dict(ar=ar, hw=hw)


    def predict(
        self,
        prompt: str = Input(description="Prompt for the video"),
        seed: int = Input(description="Seed. Leave blank to randomise", default=None),
    ) -> List[Path]:
        """Run a single prediction on the model"""
          
        # remove old output directory
        save_dir = self.cfg.save_dir
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            
        os.makedirs(save_dir, exist_ok=True)
          
        # randomize seed
        if seed is None:
            seed = random.randint(0, MAX_SEED)
            print(f"Using seed {seed}...")
        set_random_seed(seed=seed)

        samples = self.scheduler.sample(
            self.model,
            self.text_encoder,
            z_size=(self.vae.out_channels, *self.latent_size),
            prompts=[prompt],
            device=self.device,
            additional_args=self.model_args,
        )
        samples = self.vae.decode(samples.to(self.dtype))

        save_path = os.path.join(save_dir, f"output")
        save_sample(samples[0], fps=self.cfg.fps, save_path=save_path) # write file to {save_path}.mp4
                
        return [Path(f"{save_path}.mp4")]

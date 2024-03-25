# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import random
import subprocess
import numpy as np
import shutil
import time
from typing import List

from cog import BasePredictor, Input, Path

# scripts/inference.py
import os
from mmengine.config import Config


import torch
import colossalai
import torch.distributed as dist
from mmengine.runner import set_random_seed

from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import merge_args
from opensora.utils.misc import to_torch_dtype
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from colossalai.cluster import DistCoordinator

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
    # taken from 64x512x512.py
    cfg = Config(dict(
        num_frames = 64,
        fps = 24 // 2,
        image_size = (512, 512),
        dtype = "fp16",
        batch_size = 1,
        seed = 42,
        prompt_path = "./assets/texts/t2v_samples.txt",
        save_dir = "./outputs/samples/",
    ))

    # Define model
    cfg.model = dict(
        type="STDiT-XL/2",
        space_scale=1.0,
        time_scale=2 / 3,
        enable_flashattn=True,
        enable_layernorm_kernel=True,
        from_pretrained="PRETRAINED_MODEL",
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
        
        # extra config:
        ckpt_path = "pretrained_models/Open-Sora/OpenSora-v1-HQ-16x512x512.pth"
        config_file = "configs/opensora/inference/64x512x512.py"
        config_file = "configs/opensora/inference/16x512x512.py"
        
        # load config file
        # option 1: manually
        #self.cfg = cog_config()
        #self.cfg.model["from_pretrained"] = ckpt_path
        #if "multi_resolution" not in self.cfg:
        #    self.cfg["multi_resolution"] = False
        
        # command line arguments from config_utils
        extra_args = Config({
            'seed': 42,
            'ckpt_path': ckpt_path,
            'batch-size': None,
            'prompt-path': None,
            'save-dir': None,
            'num-sampling-steps': None,
            'cfg_scale': None,
        })
        
        # option 2: use config_utils
        self.cfg = Config.fromfile(config_file)
        self.cfg = merge_args(self.cfg, args=extra_args, training=False)


        # from scripts/inference

        # ======================================================
        # 1. cfg and init distributed env
        # ======================================================

        # # init distributed
        # colossalai.launch_from_torch({})
        # self.coordinator = DistCoordinator()

        # if self.coordinator.world_size > 1:
        #     set_sequence_parallel_group(dist.group.WORLD) 
        #     enable_sequence_parallelism = True
        # else:
        #     enable_sequence_parallelism = False

        # ======================================================
        # 2. runtime variables
        # ======================================================
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(self.cfg)
        self.dtype = to_torch_dtype(self.cfg.dtype)

        # ======================================================
        # 3. build model & load weights
        # ======================================================
        # 3.1. build model
        input_size = (self.cfg.num_frames, *self.cfg.image_size)
        print(f"number of frames: {self.cfg.num_frames}, image_size: {self.cfg.image_size}")
        print(f"resulting input size: {input_size}")
        self.vae = build_module(self.cfg.vae, MODELS)
        print("vae", self.vae)
        self.latent_size = self.vae.get_latent_size(input_size)
        print("latent size:", self.latent_size)
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

        # 3.2. move to device & eval
        self.vae = self.vae.to(self.device, self.dtype).eval()
        self.model = self.model.to(self.device, self.dtype).eval()

        # 3.3. build scheduler
        self.scheduler = build_module(self.cfg.scheduler, SCHEDULERS)

        # 3.4. support for multi-resolution
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

        prompts = [prompt]
          
        # ======================================================
        # 4. inference
        # ======================================================
        sample_idx = 0

        for i in range(0, len(prompts), self.cfg.batch_size):
            batch_prompts = prompts[i : i + self.cfg.batch_size]
            samples = self.scheduler.sample(
                self.model,
                self.text_encoder,
                z_size=(self.vae.out_channels, *self.latent_size),
                prompts=batch_prompts,
                device=self.device,
                additional_args=self.model_args,
            )
            samples = self.vae.decode(samples.to(self.dtype))

            save_paths = []
            # if self.coordinator.is_master():
            for idx, sample in enumerate(samples):
                print(f"Prompt: {batch_prompts[idx]}")
                save_path = os.path.join(save_dir, f"sample_{sample_idx}")
                save_sample(sample, fps=self.cfg.fps, save_path=save_path)
                save_paths.append(f"{save_path}.mp4")
                sample_idx += 1
                
        return [Path(save_path) for save_path in save_paths]

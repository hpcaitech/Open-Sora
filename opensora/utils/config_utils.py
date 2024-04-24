import argparse
import json
import os
from glob import glob

from mmengine.config import Config
from torch.utils.tensorboard import SummaryWriter


def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts


def parse_args(training=False):
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument("config", help="model config file path")

    # ======================================================
    # General
    # ======================================================
    parser.add_argument("--seed", default=42, type=int, help="generation seed")
    parser.add_argument("--ckpt-path", type=str, help="path to model ckpt; will overwrite cfg.ckpt_path if specified")
    parser.add_argument("--batch-size", default=None, type=int, help="batch size")

    # ======================================================
    # Inference
    # ======================================================
    if not training:
        # output
        parser.add_argument("--save-dir", default=None, type=str, help="path to save generated samples")
        parser.add_argument("--sample-name", default=None, type=str, help="sample name, default is sample_idx")
        parser.add_argument("--start-index", default=None, type=int, help="start index for sample name")
        parser.add_argument("--end-index", default=None, type=int, help="end index for sample name")
        parser.add_argument("--num-sample", default=None, type=int, help="number of samples to generate for one prompt")
        parser.add_argument("--prompt-as-path", action="store_true", help="use prompt as path to save samples")

        # prompt
        parser.add_argument("--prompt-path", default=None, type=str, help="path to prompt txt file")
        parser.add_argument("--prompt", default=None, type=str, nargs="+", help="prompt list")

        # image/video
        parser.add_argument("--num-frames", default=None, type=int, help="number of frames")
        parser.add_argument("--fps", default=None, type=int, help="fps")
        parser.add_argument("--image-size", default=None, type=int, nargs=2, help="image size")

        # hyperparameters
        parser.add_argument("--num-sampling-steps", default=None, type=int, help="sampling steps")
        parser.add_argument("--cfg-scale", default=None, type=float, help="balance between cond & uncond")

        # reference
        parser.add_argument("--loop", default=None, type=int, help="loop")
        parser.add_argument("--condition-frame-length", default=None, type=int, help="condition frame length")
        parser.add_argument("--reference-path", default=None, type=str, nargs="+", help="reference path")
        parser.add_argument("--mask-strategy", default=None, type=str, nargs="+", help="mask strategy")
    # ======================================================
    # Training
    # ======================================================
    else:
        parser.add_argument("--wandb", default=None, type=bool, help="enable wandb")
        parser.add_argument("--load", default=None, type=str, help="path to continue training")
        parser.add_argument("--data-path", default=None, type=str, help="path to data csv")
        parser.add_argument("--start-from-scratch", action="store_true", help="start training from scratch")

    return parser.parse_args()


def merge_args(cfg, args, training=False):
    if args.ckpt_path is not None:
        cfg.model["from_pretrained"] = args.ckpt_path
        args.ckpt_path = None
    if training and args.data_path is not None:
        cfg.dataset["data_path"] = args.data_path
        args.data_path = None
    if not training and args.cfg_scale is not None:
        cfg.scheduler["cfg_scale"] = args.cfg_scale
        args.cfg_scale = None
    if not training and args.num_sampling_steps is not None:
        cfg.scheduler["num_sampling_steps"] = args.num_sampling_steps
        args.num_sampling_steps = None

    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v

    if not training:
        # Inference only
        # - Allow not set
        if "reference_path" not in cfg:
            cfg["reference_path"] = None
        if "loop" not in cfg:
            cfg["loop"] = 1
        if "frame_interval" not in cfg:
            cfg["frame_interval"] = 1
        if "sample_name" not in cfg:
            cfg["sample_name"] = None
        if "num_sample" not in cfg:
            cfg["num_sample"] = 1
        if "prompt_as_path" not in cfg:
            cfg["prompt_as_path"] = False
        # - Prompt handling
        if "prompt" not in cfg or cfg["prompt"] is None:
            assert cfg["prompt_path"] is not None, "prompt or prompt_path must be provided"
            cfg["prompt"] = load_prompts(cfg["prompt_path"])
        if args.start_index is not None and args.end_index is not None:
            cfg["prompt"] = cfg["prompt"][args.start_index : args.end_index]
        elif args.start_index is not None:
            cfg["prompt"] = cfg["prompt"][args.start_index :]
        elif args.end_index is not None:
            cfg["prompt"] = cfg["prompt"][: args.end_index]
    else:
        # Training only
        # - Allow not set
        if "mask_ratios" not in cfg:
            cfg["mask_ratios"] = None
        if "start_from_scratch" not in cfg:
            cfg["start_from_scratch"] = False
        if "bucket_config" not in cfg:
            cfg["bucket_config"] = None
        if "transform_name" not in cfg.dataset:
            cfg.dataset["transform_name"] = "center"
        if "num_bucket_build_workers" not in cfg:
            cfg["num_bucket_build_workers"] = 1

    # Both training and inference
    if "multi_resolution" not in cfg:
        cfg["multi_resolution"] = False

    return cfg


def parse_configs(training=False):
    args = parse_args(training)
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args, training)
    return cfg


def create_experiment_workspace(cfg):
    """
    This function creates a folder for experiment tracking.

    Args:
        args: The parsed arguments.

    Returns:
        exp_dir: The path to the experiment folder.
    """
    # Make outputs folder (holds all experiment subfolders)
    os.makedirs(cfg.outputs, exist_ok=True)
    experiment_index = len(glob(f"{cfg.outputs}/*"))

    # Create an experiment folder
    model_name = cfg.model["type"].replace("/", "-")
    exp_name = f"{experiment_index:03d}-{model_name}"
    exp_dir = f"{cfg.outputs}/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    return exp_name, exp_dir


def save_training_config(cfg, experiment_dir):
    with open(f"{experiment_dir}/config.txt", "w") as f:
        json.dump(cfg, f, indent=4)


def create_tensorboard_writer(exp_dir):
    tensorboard_dir = f"{exp_dir}/tensorboard"
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    return writer

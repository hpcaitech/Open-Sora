import argparse
import json
import os
from glob import glob

from mmengine.config import Config
from torch.utils.tensorboard import SummaryWriter


def parse_args(training=False):
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument("config", help="model config file path")

    parser.add_argument("--seed", default=42, type=int, help="generation seed")
    parser.add_argument("--ckpt-path", type=str, help="path to model ckpt; will overwrite cfg.ckpt_path if specified")
    parser.add_argument("--batch-size", default=None, type=int, help="batch size")

    # ======================================================
    # Inference
    # ======================================================

    if not training:
        # prompt
        parser.add_argument("--prompt-path", default=None, type=str, help="path to prompt txt file")
        parser.add_argument("--save-dir", default=None, type=str, help="path to save generated samples")

        # hyperparameters
        parser.add_argument("--num-sampling-steps", default=None, type=int, help="sampling steps")
        parser.add_argument("--cfg-scale", default=None, type=float, help="balance between cond & uncond")
    else:
        parser.add_argument("--wandb", default=None, type=bool, help="enable wandb")
        parser.add_argument("--load", default=None, type=str, help="path to continue training")
        parser.add_argument("--data-path", default=None, type=str, help="path to data csv")

    return parser.parse_args()


def merge_args(cfg, args, training=False):
    if args.ckpt_path is not None:
        cfg.model["from_pretrained"] = args.ckpt_path
        args.ckpt_path = None

    if not training:
        if args.cfg_scale is not None:
            cfg.scheduler["cfg_scale"] = args.cfg_scale
            args.cfg_scale = None

    if "multi_resolution" not in cfg:
        cfg["multi_resolution"] = False
    for k, v in vars(args).items():
        if k in cfg and v is not None:
            cfg[k] = v

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
    exp_name = f"{experiment_index:03d}-F{cfg.num_frames}S{cfg.frame_interval}-{model_name}"
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

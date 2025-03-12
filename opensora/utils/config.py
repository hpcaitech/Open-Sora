import argparse
import ast
import json
import os
from datetime import datetime

import torch
from mmengine.config import Config

from .logger import is_distributed, is_main_process


def parse_args() -> tuple[str, argparse.Namespace]:
    """
    This function parses the command line arguments.

    Returns:
        tuple[str, argparse.Namespace]: The path to the configuration file and the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="model config file path")
    args, unknown_args = parser.parse_known_args()
    return args.config, unknown_args


def read_config(config_path: str) -> Config:
    """
    This function reads the configuration file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Config: The configuration object.
    """
    cfg = Config.fromfile(config_path)
    return cfg


def parse_configs() -> Config:
    """
    This function parses the configuration file and command line arguments.

    Returns:
        Config: The configuration object.
    """
    config, args = parse_args()
    cfg = read_config(config)
    cfg = merge_args(cfg, args)
    cfg.config_path = config

    # hard-coded for spatial compression
    if cfg.get("ae_spatial_compression", None) is not None:
        os.environ["AE_SPATIAL_COMPRESSION"] = str(cfg.ae_spatial_compression)
    return cfg


def merge_args(cfg: Config, args: argparse.Namespace) -> Config:
    """
    This function merges the configuration file and command line arguments.

    Args:
        cfg (Config): The configuration object.
        args (argparse.Namespace): The command line arguments.

    Returns:
        Config: The configuration object.
    """
    for k, v in zip(args[::2], args[1::2]):
        assert k.startswith("--"), f"Invalid argument: {k}"
        k = k[2:].replace("-", "_")
        k_split = k.split(".")
        target = cfg
        for key in k_split[:-1]:
            assert key in cfg, f"Key {key} not found in config"
            target = target[key]
        if v.lower() == "none":
            v = None
        elif k in target:
            v_type = type(target[k])
            if v_type == bool:
                v = auto_convert(v)
            else:
                v = type(target[k])(v)
        else:
            v = auto_convert(v)
        target[k_split[-1]] = v
    return cfg


def auto_convert(value: str) -> int | float | bool | list | dict | None:
    """
    Automatically convert a string to the appropriate Python data type,
    including int, float, bool, list, dict, etc.

    Args:
        value (str): The string to convert.

    Returns:
        int, float, bool, list |  dict: The converted value.
    """
    # Handle empty string
    if value == "":
        return value

    # Handle None
    if value.lower() == "none":
        return None

    # Handle boolean values
    lower_value = value.lower()
    if lower_value == "true":
        return True
    elif lower_value == "false":
        return False

    # Try to convert the string to an integer or float
    try:
        # Try converting to an integer
        return int(value)
    except ValueError:
        pass

    try:
        # Try converting to a float
        return float(value)
    except ValueError:
        pass

    # Try to convert the string to a list, dict, tuple, etc.
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass

    # If all attempts fail, return the original string
    return value


def sync_string(value: str):
    """
    This function synchronizes a string across all processes.
    """
    if not is_distributed():
        return value
    bytes_value = value.encode("utf-8")
    max_len = 256
    bytes_tensor = torch.zeros(max_len, dtype=torch.uint8).cuda()
    bytes_tensor[: len(bytes_value)] = torch.tensor(
        list(bytes_value), dtype=torch.uint8
    )
    torch.distributed.broadcast(bytes_tensor, 0)
    synced_value = bytes_tensor.cpu().numpy().tobytes().decode("utf-8").rstrip("\x00")
    return synced_value


def create_experiment_workspace(
    output_dir: str, model_name: str = None, config: dict = None, exp_name: str = None
) -> tuple[str, str]:
    """
    This function creates a folder for experiment tracking.

    Args:
        output_dir: The path to the output directory.
        model_name: The name of the model.
        exp_name: The given name of the experiment, if None will use default.

    Returns:
        tuple[str, str]: The experiment name and the experiment directory.
    """
    if exp_name is None:
        # Make outputs folder (holds all experiment subfolders)
        experiment_index = datetime.now().strftime("%y%m%d_%H%M%S")
        experiment_index = sync_string(experiment_index)
        # Create an experiment folder
        model_name = (
            "-" + model_name.replace("/", "-") if model_name is not None else ""
        )
        exp_name = f"{experiment_index}{model_name}"
    exp_dir = f"{output_dir}/{exp_name}"
    if is_main_process():
        os.makedirs(exp_dir, exist_ok=True)
        # Save the config
        with open(f"{exp_dir}/config.txt", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    return exp_name, exp_dir


def config_to_name(cfg: Config) -> str:
    filename = cfg._filename
    filename = filename.replace("configs/", "")
    filename = filename.replace(".py", "")
    filename = filename.replace("/", "_")
    return filename


def parse_alias(cfg: Config) -> Config:
    if cfg.get("resolution", None) is not None:
        cfg.sampling_option.resolution = cfg.resolution
    if cfg.get("guidance", None) is not None:
        cfg.sampling_option.guidance = float(cfg.guidance)
    if cfg.get("guidance_img", None) is not None:
        cfg.sampling_option.guidance_img = float(cfg.guidance_img)
    if cfg.get("num_steps", None) is not None:
        cfg.sampling_option.num_steps = int(cfg.num_steps)
    if cfg.get("num_frames", None) is not None:
        cfg.sampling_option.num_frames = int(cfg.num_frames)
    if cfg.get("aspect_ratio", None) is not None:
        cfg.sampling_option.aspect_ratio = cfg.aspect_ratio
    if cfg.get("ckpt_path", None) is not None:
        cfg.model.from_pretrained = cfg.ckpt_path
    return cfg

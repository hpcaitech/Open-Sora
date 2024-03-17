import functools
import json
import logging
import operator
import os
from typing import Tuple

import colossalai
import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.booster import Booster
from colossalai.checkpoint_io import GeneralCheckpointIO
from colossalai.cluster import DistCoordinator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.datasets.utils import download_url

pretrained_models = {
    "DiT-XL-2-512x512.pt": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt",
    "DiT-XL-2-256x256.pt": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt",
    "Latte-XL-2-256x256-ucf101.pt": "https://huggingface.co/maxin-cn/Latte/resolve/main/ucf101.pt",
    "PixArt-XL-2-256x256.pth": "https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-256x256.pth",
    "PixArt-XL-2-SAM-256x256.pth": "https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-SAM-256x256.pth",
    "PixArt-XL-2-512x512.pth": "https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-512x512.pth",
    "PixArt-XL-2-1024-MS.pth": "https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-1024-MS.pth",
}


def reparameter(ckpt, name=None):
    if "DiT" in name:
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
    elif "Latte" in name:
        ckpt = ckpt["ema"]
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
        del ckpt["temp_embed"]
    elif "PixArt" in name:
        ckpt = ckpt["state_dict"]
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
    return ckpt


def find_model(model_name):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        model = download_model(model_name)
        model = reparameter(model, model_name)
        return model
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(model_name), f"Could not find DiT checkpoint at {model_name}"
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        if "pos_embed_temporal" in checkpoint:
            del checkpoint["pos_embed_temporal"]
        if "pos_embed" in checkpoint:
            del checkpoint["pos_embed"]
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
        return checkpoint


def download_model(model_name):
    """
    Downloads a pre-trained DiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f"pretrained_models/{model_name}"
    if not os.path.isfile(local_path):
        os.makedirs("pretrained_models", exist_ok=True)
        web_path = pretrained_models[model_name]
        download_url(web_path, "pretrained_models", model_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


def load_from_sharded_state_dict(model, ckpt_path):
    ckpt_io = GeneralCheckpointIO()
    ckpt_io.load_model(model, os.path.join(ckpt_path, "model"))

def model_sharding(model: torch.nn.Module):
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    for _, param in model.named_parameters():
        padding_size = (world_size - param.numel() % world_size) % world_size
        if padding_size > 0:
            padding_param = torch.nn.functional.pad(param.data.view(-1), [0, padding_size])
        else:
            padding_param = param.data.view(-1)
        splited_params = padding_param.split(padding_param.numel() // world_size)
        splited_params = splited_params[global_rank]
        param.data = splited_params


def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def remove_padding(tensor: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
    return tensor[: functools.reduce(operator.mul, original_shape)]


def model_gathering(model: torch.nn.Module, model_shape_dict: dict):
    global_rank = dist.get_rank()
    global_size = dist.get_world_size()
    for name, param in model.named_parameters():
        all_params = [torch.empty_like(param.data) for _ in range(global_size)]
        dist.all_gather(all_params, param.data, group=dist.group.WORLD)
        if int(global_rank) == 0:
            all_params = torch.cat(all_params)
            param.data = remove_padding(all_params, model_shape_dict[name]).view(model_shape_dict[name])
    dist.barrier()


def record_model_param_shape(model: torch.nn.Module) -> dict:
    param_shape = {}
    for name, param in model.named_parameters():
        param_shape[name] = param.shape
    return param_shape


def save(
    booster: Booster,
    model: nn.Module,
    ema: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    epoch: int,
    step: int,
    global_step: int,
    batch_size: int,
    coordinator: DistCoordinator,
    save_dir: str,
    shape_dict: dict,
):
    save_dir = os.path.join(save_dir, f"epoch{epoch}-global_step{global_step}")
    os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)

    booster.save_model(model, os.path.join(save_dir, "model"), shard=True)
    # ema is not boosted, so we don't need to use booster.save_model
    model_gathering(ema, shape_dict)
    global_rank = dist.get_rank()
    if int(global_rank) == 0:
        torch.save(ema.state_dict(), os.path.join(save_dir, "ema.pt"))
        model_sharding(ema)

    booster.save_optimizer(optimizer, os.path.join(save_dir, "optimizer"), shard=True, size_per_shard=4096)
    if lr_scheduler is not None:
        booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
    running_states = {
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "sample_start_index": step * batch_size,
    }
    if coordinator.is_master():
        save_json(running_states, os.path.join(save_dir, "running_states.json"))
    dist.barrier()


def load(
    booster: Booster, model: nn.Module, ema: nn.Module, optimizer: Optimizer, lr_scheduler: _LRScheduler, load_dir: str
) -> Tuple[int, int, int]:
    booster.load_model(model, os.path.join(load_dir, "model"))
    # ema is not boosted, so we don't use booster.load_model
    # ema.load_state_dict(torch.load(os.path.join(load_dir, "ema.pt")))
    ema.load_state_dict(torch.load(os.path.join(load_dir, "ema.pt"), map_location=torch.device("cpu")))
    booster.load_optimizer(optimizer, os.path.join(load_dir, "optimizer"))
    if lr_scheduler is not None:
        booster.load_lr_scheduler(lr_scheduler, os.path.join(load_dir, "lr_scheduler"))
    running_states = load_json(os.path.join(load_dir, "running_states.json"))
    dist.barrier()
    return running_states["epoch"], running_states["step"], running_states["sample_start_index"]


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def load_checkpoint(model, ckpt_path, save_as_pt=True):
    if ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        state_dict = find_model(ckpt_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    elif os.path.isdir(ckpt_path):
        load_from_sharded_state_dict(model, ckpt_path)
        if save_as_pt:
            save_path = os.path.join(ckpt_path, "model_ckpt.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved to {save_path}")
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")

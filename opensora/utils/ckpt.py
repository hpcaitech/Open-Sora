import functools
import json
import operator
import os
import re
import shutil
from glob import glob
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.booster import Booster
from colossalai.checkpoint_io import GeneralCheckpointIO
from colossalai.utils.safetensors import save as async_save
from colossalai.zero.low_level import LowLevelZeroOptimizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tensornvme.async_file_io import AsyncFileWriter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from opensora.acceleration.parallel_states import get_data_parallel_group

from .logger import log_message

hf_endpoint = os.environ.get("HF_ENDPOINT")
if hf_endpoint is None:
    hf_endpoint = "https://huggingface.co"
os.environ["TENSORNVME_DEBUG"] = "1"


def load_from_hf_hub(repo_path: str, cache_dir: str = None) -> str:
    """
    Loads a checkpoint from the Hugging Face Hub.

    Args:
        repo_path (str): The path to the checkpoint on the Hugging Face Hub.
        cache_dir (str): The directory to cache the downloaded checkpoint.

    Returns:
        str: The path to the downloaded checkpoint.
    """
    repo_id = "/".join(repo_path.split("/")[:-1])
    repo_file = repo_path.split("/")[-1]
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=repo_file, cache_dir=cache_dir)
    return ckpt_path


def load_from_sharded_state_dict(model: nn.Module, ckpt_path: str, model_name: str = "model", strict=False):
    """
    Loads a model from a sharded checkpoint.

    Args:
        model (nn.Module): The model to load the checkpoint into.
        ckpt_path (str): The path to the checkpoint.
        model_name (str): The name of the model in the checkpoint.
        strict (bool): Whether to strictly enforce that the keys in the checkpoint match the keys in the model.
    """
    ckpt_io = GeneralCheckpointIO()
    ckpt_io.load_model(model, os.path.join(ckpt_path, model_name), strict=strict)


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    """
    Prints a warning if there are missing or unexpected keys when loading a model.

    Args:
        missing (list[str]): The missing keys.
        unexpected (list[str]): The unexpected keys.
    """
    if len(missing) > 0 and len(unexpected) > 0:
        log_message(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        log_message("\n" + "-" * 79 + "\n")
        log_message(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        log_message(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        log_message(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    else:
        log_message("Model loaded successfully")


def load_checkpoint(
    model: nn.Module,
    path: str,
    cache_dir: str = None,
    device_map: torch.device | str = "cpu",
    cai_model_name: str = "model",
    strict: bool = False,
    rename_keys: dict = None,  # rename keys in the checkpoint to support fine-tuning with a different model architecture; map old_key_prefix to new_key_prefix
) -> nn.Module:
    """
    Loads a checkpoint into model from a path. Support three types of checkpoints:
        1. huggingface safetensors
        2. local .pt or .pth
        3. colossalai sharded checkpoint

    Args:
        model (nn.Module): The model to load the checkpoint into.
        path (str): The path to the checkpoint.
        cache_dir (str): The directory to cache the downloaded checkpoint.
        device_map (torch.device | str): The device to map the checkpoint to.
        cai_model_name (str): The name of the model in the checkpoint.

    Returns:
        nn.Module: The model with the loaded checkpoint.
    """
    if not os.path.exists(path):
        log_message(f"Checkpoint not found at {path}, trying to download from Hugging Face Hub")
        path = load_from_hf_hub(path, cache_dir)
    assert os.path.exists(path), f"Could not find checkpoint at {path}"

    log_message(f"Loading checkpoint from {path}")
    if path.endswith(".safetensors"):
        ckpt = load_file(path, device='cpu')

        if rename_keys is not None:
            # rename keys in the loaded state_dict with old_key_prefix to with new_key_prefix.
            renamed_ckpt = {}
            for old_key, v in ckpt.items():
                new_key = old_key
                for old_key_prefix, new_key_prefix in rename_keys.items():
                    if old_key_prefix in old_key:
                        new_key = old_key.replace(old_key_prefix, new_key_prefix)
                        print(f"Renamed {old_key} to {new_key} in the loaded state_dict")
                        break
                renamed_ckpt[new_key] = v
            ckpt = renamed_ckpt

        missing, unexpected = model.load_state_dict(ckpt, strict=strict)
        print_load_warning(missing, unexpected)
    elif path.endswith(".pt") or path.endswith(".pth"):
        ckpt = torch.load(path, map_location=device_map)
        missing, unexpected = model.load_state_dict(ckpt, strict=strict)
        print_load_warning(missing, unexpected)
    else:
        assert os.path.isdir(path), f"Invalid checkpoint path: {path}"
        load_from_sharded_state_dict(model, path, model_name=cai_model_name, strict=strict)
    return model


def rm_checkpoints(
    save_dir: str,
    keep_n_latest: int = 0,
):
    """
    Remove old checkpoints.

    Args:
        save_dir (str): The directory to save the checkpoints.
        keep_n_latest (int): The number of latest checkpoints to keep.
    """
    if keep_n_latest <= 0 or dist.get_rank() != 0:
        return
    files = glob(os.path.join(save_dir, "epoch*-global_step*"))
    files = sorted(
        files, key=lambda s: tuple(map(int, re.search(r"epoch(\d+)-global_step(\d+)", s).groups())), reverse=True
    )
    to_remove = files[keep_n_latest:]
    for f in to_remove:
        # shutil.rmtree(f)
        for item in glob(os.path.join(f, "*")):
            if os.path.isdir(item):
                dir_name = os.path.basename(item)
                if dir_name != "eval":
                    shutil.rmtree(item)
            else:
                os.remove(item)


def model_sharding(model: torch.nn.Module, device: torch.device = None):
    """
    Sharding the model parameters across multiple GPUs.

    Args:
        model (torch.nn.Module): The model to shard.
        device (torch.device): The device to shard the model to.
    """
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    for _, param in model.named_parameters():
        if device is None:
            device = param.device
        padding_size = (world_size - param.numel() % world_size) % world_size
        if padding_size > 0:
            padding_param = torch.nn.functional.pad(param.data.view(-1), [0, padding_size])
        else:
            padding_param = param.data.view(-1)
        splited_params = padding_param.split(padding_param.numel() // world_size)
        splited_params = splited_params[global_rank]
        param.data = splited_params.to(device)


def model_gathering(model: torch.nn.Module, model_shape_dict: dict, pinned_state_dict: dict) -> None:
    """
    Gather the model parameters from multiple GPUs.

    Args:
        model (torch.nn.Module): The model to gather.
        model_shape_dict (dict): The shape of the model parameters.
        device (torch.device): The device to gather the model to.
    """
    global_rank = dist.get_rank()
    global_size = dist.get_world_size()
    params = set()
    for name, param in model.named_parameters():
        params.add(name)
        all_params = [torch.empty_like(param.data) for _ in range(global_size)]
        dist.all_gather(all_params, param.data, group=dist.group.WORLD)
        if int(global_rank) == 0:
            all_params = torch.cat(all_params)
            gathered_param = remove_padding(all_params, model_shape_dict[name]).view(model_shape_dict[name])
            pinned_state_dict[name].copy_(gathered_param)
    if int(global_rank) == 0:
        for k, v in model.state_dict(keep_vars=True).items():
            if k not in params:
                pinned_state_dict[k].copy_(v)

    dist.barrier()


def remove_padding(tensor: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """
    Remove padding from a tensor.

    Args:
        tensor (torch.Tensor): The tensor to remove padding from.
        original_shape (tuple): The original shape of the tensor.
    """
    return tensor[: functools.reduce(operator.mul, original_shape)]


def record_model_param_shape(model: torch.nn.Module) -> dict:
    """
    Record the shape of the model parameters.

    Args:
        model (torch.nn.Module): The model to record the parameter shape of.

    Returns:
        dict: The shape of the model parameters.
    """
    param_shape = {}
    for name, param in model.named_parameters():
        param_shape[name] = param.shape
    return param_shape


def load_json(file_path: str) -> dict:
    """
    Load a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The loaded JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, file_path: str):
    """
    Save a dictionary to a JSON file.

    Args:
        data: The dictionary to save.
        file_path (str): The path to save the JSON file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _prepare_ema_pinned_state_dict(model: nn.Module, ema_shape_dict: dict):
    ema_pinned_state_dict = dict()
    for name, p in model.named_parameters():
        ema_pinned_state_dict[name] = torch.empty(ema_shape_dict[name], pin_memory=True, device="cpu", dtype=p.dtype)
    sd = model.state_dict(keep_vars=True)
    # handle buffers
    for k, v in sd.items():
        if k not in ema_pinned_state_dict:
            ema_pinned_state_dict[k] = torch.empty(v.shape, pin_memory=True, device="cpu", dtype=v.dtype)

    return ema_pinned_state_dict


def _search_valid_path(path: str) -> str:
    if os.path.exists(f"{path}.safetensors"):
        return f"{path}.safetensors"
    elif os.path.exists(f"{path}.pt"):
        return f"{path}.pt"
    return path


def master_weights_gathering(model: torch.nn.Module, optimizer: LowLevelZeroOptimizer, pinned_state_dict: dict) -> None:
    """
    Gather the model parameters from multiple GPUs.

    Args:
        model (torch.nn.Module): The model to gather.
        model_shape_dict (dict): The shape of the model parameters.
        device (torch.device): The device to gather the model to.
    """
    w2m = optimizer.get_working_to_master_map()
    for name, param in model.named_parameters():
        master_p = w2m[id(param)]
        zero_pg = optimizer.param_to_pg[param]
        world_size = dist.get_world_size(zero_pg)
        all_params = [torch.empty_like(master_p) for _ in range(world_size)]
        dist.all_gather(all_params, master_p, group=zero_pg)
        if dist.get_rank() == 0:
            all_params = torch.cat(all_params)
            gathered_param = remove_padding(all_params, param.shape).view(param.shape)
            pinned_state_dict[name].copy_(gathered_param)

    dist.barrier()


def load_master_weights(model: torch.nn.Module, optimizer: LowLevelZeroOptimizer, state_dict: dict) -> None:
    pg = get_data_parallel_group(get_mixed_dp_pg=True)
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    w2m = optimizer.get_working_to_master_map()
    for name, param in model.named_parameters():
        master_p = w2m[id(param)]
        state = state_dict[name].view(-1)
        padding_size = len(master_p) * world_size - len(state)
        state = torch.nn.functional.pad(state, [0, padding_size])
        target_chunk = state.chunk(world_size)[rank].to(master_p.dtype)
        master_p[: len(target_chunk)].copy_(target_chunk)


class CheckpointIO:
    def __init__(self, n_write_entries: int = 32):
        self.n_write_entries = n_write_entries
        self.writer: Optional[AsyncFileWriter] = None
        self.pinned_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.master_pinned_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.master_writer: Optional[AsyncFileWriter] = None

    def _sync_io(self):
        if self.writer is not None:
            self.writer.synchronize()
            self.writer = None
        if self.master_writer is not None:
            self.master_writer.synchronize()
            self.master_writer = None

    def __del__(self):
        self._sync_io()

    def _prepare_pinned_state_dict(self, ema: nn.Module, ema_shape_dict: dict):
        if self.pinned_state_dict is None and dist.get_rank() == 0:
            self.pinned_state_dict = _prepare_ema_pinned_state_dict(ema, ema_shape_dict)

    def _prepare_master_pinned_state_dict(self, model: nn.Module, optimizer: LowLevelZeroOptimizer):
        if self.master_pinned_state_dict is None and dist.get_rank() == 0:
            sd = {}
            w2m = optimizer.get_working_to_master_map()
            for n, p in model.named_parameters():
                master_p = w2m[id(p)]
                sd[n] = torch.empty(p.shape, dtype=master_p.dtype, pin_memory=True, device="cpu")
            self.master_pinned_state_dict = sd

    def save(
        self,
        booster: Booster,
        save_dir: str,
        model: nn.Module = None,
        ema: nn.Module = None,
        optimizer: Optimizer = None,
        lr_scheduler: _LRScheduler = None,
        sampler=None,
        epoch: int = None,
        step: int = None,
        global_step: int = None,
        batch_size: int = None,
        lora: bool = False,
        actual_update_step: int = None,
        ema_shape_dict: dict = None,
        async_io: bool = True,
        include_master_weights: bool = False,
    ) -> str:
        """
        Save a checkpoint.

        Args:
            booster (Booster): The Booster object.
            save_dir (str): The directory to save the checkpoint to.
            model (nn.Module): The model to save the checkpoint from.
            ema (nn.Module): The EMA model to save the checkpoint from.
            optimizer (Optimizer): The optimizer to save the checkpoint from.
            lr_scheduler (_LRScheduler): The learning rate scheduler to save the checkpoint from.
            sampler: The sampler to save the checkpoint from.
            epoch (int): The epoch of the checkpoint.
            step (int): The step of the checkpoint.
            global_step (int): The global step of the checkpoint.
            batch_size (int): The batch size of the checkpoint.
            lora (bool): Whether the model is trained with LoRA.

        Returns:
            str: The path to the saved checkpoint
        """
        self._sync_io()
        save_dir = os.path.join(save_dir, f"epoch{epoch}-global_step{actual_update_step}")
        os.environ["TENSORNVME_DEBUG_LOG"] = os.path.join(save_dir, "async_file_io.log")
        if model is not None:
            if not lora:
                os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)
                booster.save_model(
                    model,
                    os.path.join(save_dir, "model"),
                    shard=True,
                    use_safetensors=True,
                    size_per_shard=4096,
                    use_async=async_io,
                )
            else:
                os.makedirs(os.path.join(save_dir, "lora"), exist_ok=True)
                booster.save_lora_as_pretrained(model, os.path.join(save_dir, "lora"))
        if optimizer is not None:
            booster.save_optimizer(
                optimizer, os.path.join(save_dir, "optimizer"), shard=True, size_per_shard=4096, use_async=async_io
            )
            if include_master_weights:
                self._prepare_master_pinned_state_dict(model, optimizer)
                master_weights_gathering(model, optimizer, self.master_pinned_state_dict)
        if lr_scheduler is not None:
            booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
        if ema is not None:
            self._prepare_pinned_state_dict(ema, ema_shape_dict)
            model_gathering(ema, ema_shape_dict, self.pinned_state_dict)
        if dist.get_rank() == 0:
            running_states = {
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "batch_size": batch_size,
                "actual_update_step": actual_update_step,
            }
            save_json(running_states, os.path.join(save_dir, "running_states.json"))

            if ema is not None:
                if async_io:
                    self.writer = async_save(os.path.join(save_dir, "ema.safetensors"), self.pinned_state_dict)
                else:
                    torch.save(ema.state_dict(), os.path.join(save_dir, "ema.pt"))

            if sampler is not None:
                # only for VariableVideoBatchSampler
                torch.save(sampler.state_dict(step), os.path.join(save_dir, "sampler"))

            if optimizer is not None and include_master_weights:
                self.master_writer = async_save(
                    os.path.join(save_dir, "master.safetensors"), self.master_pinned_state_dict
                )

        dist.barrier()
        return save_dir

    def load(
        self,
        booster: Booster,
        load_dir: str,
        model: nn.Module = None,
        ema: nn.Module = None,
        optimizer: Optimizer = None,
        lr_scheduler: _LRScheduler = None,
        sampler=None,
        strict: bool = False,
        include_master_weights: bool = False,
    ) -> tuple[int, int]:
        """
        Load a checkpoint.

        Args:
            booster (Booster): The Booster object.
            load_dir (str): The directory to load the checkpoint from.
            model (nn.Module): The model to load the checkpoint into.
            ema (nn.Module): The EMA model to load the checkpoint into.
            optimizer (Optimizer): The optimizer to load the checkpoint into.
            lr_scheduler (_LRScheduler): The learning rate scheduler to load the checkpoint into.
            sampler: The sampler to load the checkpoint into.

        Returns:
            tuple[int, int]: The epoch and step of the checkpoint.
        """
        assert os.path.exists(load_dir), f"Checkpoint directory {load_dir} does not exist"
        assert os.path.exists(os.path.join(load_dir, "running_states.json")), "running_states.json does not exist"

        running_states = load_json(os.path.join(load_dir, "running_states.json"))
        if model is not None:
            booster.load_model(
                model,
                _search_valid_path(os.path.join(load_dir, "model")),
                strict=strict,
                low_cpu_mem_mode=False,
                num_threads=32,
            )
        if ema is not None:
            if os.path.exists(os.path.join(load_dir, "ema.safetensors")):
                ema_state_dict = load_file(os.path.join(load_dir, "ema.safetensors"))
            else:
                ema_state_dict = torch.load(os.path.join(load_dir, "ema.pt"), map_location=torch.device("cpu"))
            # ema is not boosted, so we don't use booster.load_model
            ema.load_state_dict(ema_state_dict, strict=strict, assign=True)

        if optimizer is not None:
            booster.load_optimizer(
                optimizer, os.path.join(load_dir, "optimizer"), low_cpu_mem_mode=False, num_threads=32
            )
            if include_master_weights:
                master_state_dict = load_file(os.path.join(load_dir, "master.safetensors"))
                load_master_weights(model, optimizer, master_state_dict)
        if lr_scheduler is not None:
            booster.load_lr_scheduler(lr_scheduler, os.path.join(load_dir, "lr_scheduler"))
        if sampler is not None:
            sampler.load_state_dict(torch.load(os.path.join(load_dir, "sampler")))

        dist.barrier()

        return (running_states["epoch"], running_states["step"])

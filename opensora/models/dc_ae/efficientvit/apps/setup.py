import os
import time
from copy import deepcopy
from typing import Optional

import torch.backends.cudnn
import torch.distributed
import torch.nn as nn

from ..apps.utils import (
    dist_init,
    dump_config,
    get_dist_local_rank,
    get_dist_rank,
    init_modules,
    is_master,
    load_config,
    partial_update_config,
    zero_last_gamma,
)
from ..models.utils import load_state_dict_from_file

__all__ = [
    "save_exp_config",
    "setup_dist_env",
    "setup_seed",
    "setup_exp_config",
    "init_model",
]


def save_exp_config(exp_config: dict, path: str, name="config.yaml") -> None:
    if not is_master():
        return
    dump_config(exp_config, os.path.join(path, name))


def setup_dist_env(gpu: Optional[str] = None) -> None:
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    if not torch.distributed.is_initialized():
        dist_init()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(get_dist_local_rank())


def setup_seed(manual_seed: int, resume: bool) -> None:
    if resume:
        manual_seed = int(time.time())
    manual_seed = get_dist_rank() + manual_seed
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)


def setup_exp_config(config_path: str, recursive=True, opt_args: Optional[dict] = None) -> dict:
    # load config
    if not os.path.isfile(config_path):
        raise ValueError(config_path)

    fpaths = [config_path]
    if recursive:
        extension = os.path.splitext(config_path)[1]
        while os.path.dirname(config_path) != config_path:
            config_path = os.path.dirname(config_path)
            fpath = os.path.join(config_path, "default" + extension)
            if os.path.isfile(fpath):
                fpaths.append(fpath)
        fpaths = fpaths[::-1]

    default_config = load_config(fpaths[0])
    exp_config = deepcopy(default_config)
    for fpath in fpaths[1:]:
        partial_update_config(exp_config, load_config(fpath))
    # update config via args
    if opt_args is not None:
        partial_update_config(exp_config, opt_args)

    return exp_config


def init_model(
    network: nn.Module,
    init_from: Optional[str] = None,
    backbone_init_from: Optional[str] = None,
    rand_init="trunc_normal",
    last_gamma=None,
) -> None:
    # initialization
    init_modules(network, init_type=rand_init)
    # zero gamma of last bn in each block
    if last_gamma is not None:
        zero_last_gamma(network, last_gamma)

    # load weight
    if init_from is not None and os.path.isfile(init_from):
        network.load_state_dict(load_state_dict_from_file(init_from))
        print(f"Loaded init from {init_from}")
    elif backbone_init_from is not None and os.path.isfile(backbone_init_from):
        network.backbone.load_state_dict(load_state_dict_from_file(backbone_init_from))
        print(f"Loaded backbone init from {backbone_init_from}")
    else:
        print(f"Random init ({rand_init}) with last gamma {last_gamma}")

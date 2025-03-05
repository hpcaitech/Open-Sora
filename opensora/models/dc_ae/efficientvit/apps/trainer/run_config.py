# Copyright 2024 MIT Han Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import numpy as np
import torch.nn as nn

from ...apps.utils import CosineLRwithWarmup, build_optimizer

__all__ = ["Scheduler", "RunConfig"]


class Scheduler:
    PROGRESS = 0


class RunConfig:
    n_epochs: int
    init_lr: float
    warmup_epochs: int
    warmup_lr: float
    lr_schedule_name: str
    lr_schedule_param: dict
    optimizer_name: str
    optimizer_params: dict
    weight_decay: float
    no_wd_keys: list
    grad_clip: float  # allow none to turn off grad clipping
    reset_bn: bool
    reset_bn_size: int
    reset_bn_batch_size: int
    eval_image_size: list  # allow none to use image_size in data_provider

    @property
    def none_allowed(self):
        return ["grad_clip", "eval_image_size"]

    def __init__(self, **kwargs):  # arguments must be passed as kwargs
        for k, val in kwargs.items():
            setattr(self, k, val)

        # check that all relevant configs are there
        annotations = {}
        for clas in type(self).mro():
            if hasattr(clas, "__annotations__"):
                annotations.update(clas.__annotations__)
        for k, k_type in annotations.items():
            assert hasattr(self, k), f"Key {k} with type {k_type} required for initialization."
            attr = getattr(self, k)
            if k in self.none_allowed:
                k_type = (k_type, type(None))
            assert isinstance(attr, k_type), f"Key {k} must be type {k_type}, provided={attr}."

        self.global_step = 0
        self.batch_per_epoch = 1

    def build_optimizer(self, network: nn.Module) -> tuple[Any, Any]:
        r"""require setting 'batch_per_epoch' before building optimizer & lr_scheduler"""
        param_dict = {}
        for name, param in network.named_parameters():
            if param.requires_grad:
                opt_config = [self.weight_decay, self.init_lr]
                if self.no_wd_keys is not None and len(self.no_wd_keys) > 0:
                    if np.any([key in name for key in self.no_wd_keys]):
                        opt_config[0] = 0
                opt_key = json.dumps(opt_config)
                param_dict[opt_key] = param_dict.get(opt_key, []) + [param]

        net_params = []
        for opt_key, param_list in param_dict.items():
            wd, lr = json.loads(opt_key)
            net_params.append({"params": param_list, "weight_decay": wd, "lr": lr})

        optimizer = build_optimizer(net_params, self.optimizer_name, self.optimizer_params, self.init_lr)
        # build lr scheduler
        if self.lr_schedule_name == "cosine":
            decay_steps = []
            for epoch in self.lr_schedule_param.get("step", []):
                decay_steps.append(epoch * self.batch_per_epoch)
            decay_steps.append(self.n_epochs * self.batch_per_epoch)
            decay_steps.sort()
            lr_scheduler = CosineLRwithWarmup(
                optimizer,
                self.warmup_epochs * self.batch_per_epoch,
                self.warmup_lr,
                decay_steps,
            )
        else:
            raise NotImplementedError
        return optimizer, lr_scheduler

    def update_global_step(self, epoch, batch_id=0) -> None:
        self.global_step = epoch * self.batch_per_epoch + batch_id
        Scheduler.PROGRESS = self.progress

    @property
    def progress(self) -> float:
        warmup_steps = self.warmup_epochs * self.batch_per_epoch
        steps = max(0, self.global_step - warmup_steps)
        return steps / (self.n_epochs * self.batch_per_epoch)

    def step(self) -> None:
        self.global_step += 1
        Scheduler.PROGRESS = self.progress

    def get_remaining_epoch(self, epoch, post=True) -> int:
        return self.n_epochs + self.warmup_epochs - epoch - int(post)

    def epoch_format(self, epoch: int) -> str:
        epoch_format = f"%.{len(str(self.n_epochs))}d"
        epoch_format = f"[{epoch_format}/{epoch_format}]"
        epoch_format = epoch_format % (epoch + 1 - self.warmup_epochs, self.n_epochs)
        return epoch_format

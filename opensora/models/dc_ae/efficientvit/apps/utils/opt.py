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

from typing import Any, Optional

import torch

__all__ = ["REGISTERED_OPTIMIZER_DICT", "build_optimizer"]

# register optimizer here
#   name: optimizer, kwargs with default values
REGISTERED_OPTIMIZER_DICT: dict[str, tuple[type, dict[str, Any]]] = {
    "sgd": (torch.optim.SGD, {"momentum": 0.9, "nesterov": True}),
    "adam": (torch.optim.Adam, {"betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False}),
    "adamw": (torch.optim.AdamW, {"betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False}),
}


def build_optimizer(
    net_params, optimizer_name: str, optimizer_params: Optional[dict], init_lr: float
) -> torch.optim.Optimizer:
    optimizer_class, default_params = REGISTERED_OPTIMIZER_DICT[optimizer_name]
    optimizer_params = {} if optimizer_params is None else optimizer_params

    for key in default_params:
        if key in optimizer_params:
            default_params[key] = optimizer_params[key]
    optimizer = optimizer_class(net_params, init_lr, **default_params)
    return optimizer

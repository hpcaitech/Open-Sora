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

from typing import Union

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ["init_modules"]


def init_modules(model: Union[nn.Module, list[nn.Module]], init_type="trunc_normal") -> None:
    _DEFAULT_INIT_PARAM = {"trunc_normal": 0.02}

    if isinstance(model, list):
        for sub_module in model:
            init_modules(sub_module, init_type)
    else:
        init_params = init_type.split("@")
        init_params = float(init_params[1]) if len(init_params) > 1 else None

        if init_type.startswith("trunc_normal"):
            init_func = lambda param: nn.init.trunc_normal_(
                param, std=(_DEFAULT_INIT_PARAM["trunc_normal"] if init_params is None else init_params)
            )
        elif init_type.startswith("normal"):
            init_func = lambda param: nn.init.normal_(
                param, std=(_DEFAULT_INIT_PARAM["trunc_normal"] if init_params is None else init_params)
            )
        else:
            raise NotImplementedError

        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                init_func(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                init_func(m.weight)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            else:
                weight = getattr(m, "weight", None)
                bias = getattr(m, "bias", None)
                if isinstance(weight, torch.nn.Parameter):
                    init_func(weight)
                if isinstance(bias, torch.nn.Parameter):
                    bias.data.zero_()
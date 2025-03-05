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

import copy
import math

import torch
import torch.nn as nn

from ...models.utils import is_parallel

__all__ = ["EMA"]


def update_ema(ema: nn.Module, new_state_dict: dict[str, torch.Tensor], decay: float) -> None:
    for k, v in ema.state_dict().items():
        if v.dtype.is_floating_point:
            v -= (1.0 - decay) * (v - new_state_dict[k].detach())


class EMA:
    def __init__(self, model: nn.Module, decay: float, warmup_steps=2000):
        self.shadows = copy.deepcopy(model.module if is_parallel(model) else model).eval()
        self.decay = decay
        self.warmup_steps = warmup_steps

        for p in self.shadows.parameters():
            p.requires_grad = False

    def step(self, model: nn.Module, global_step: int) -> None:
        with torch.no_grad():
            msd = (model.module if is_parallel(model) else model).state_dict()
            update_ema(self.shadows, msd, self.decay * (1 - math.exp(-global_step / self.warmup_steps)))

    def state_dict(self) -> dict[float, dict[str, torch.Tensor]]:
        return {self.decay: self.shadows.state_dict()}

    def load_state_dict(self, state_dict: dict[float, dict[str, torch.Tensor]]) -> None:
        for decay in state_dict:
            if decay == self.decay:
                self.shadows.load_state_dict(state_dict[decay])

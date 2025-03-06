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

import math
from typing import Union

import torch

from ...models.utils.list import val2list

__all__ = ["CosineLRwithWarmup", "ConstantLRwithWarmup"]


class CosineLRwithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        warmup_lr: float,
        decay_steps: Union[int, list[int]],
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.decay_steps = val2list(decay_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_steps:
            return [
                (base_lr - self.warmup_lr) * (self.last_epoch + 1) / self.warmup_steps + self.warmup_lr
                for base_lr in self.base_lrs
            ]
        else:
            current_steps = self.last_epoch - self.warmup_steps
            decay_steps = [0] + self.decay_steps
            idx = len(decay_steps) - 2
            for i, decay_step in enumerate(decay_steps[:-1]):
                if decay_step <= current_steps < decay_steps[i + 1]:
                    idx = i
                    break
            current_steps -= decay_steps[idx]
            decay_step = decay_steps[idx + 1] - decay_steps[idx]
            return [0.5 * base_lr * (1 + math.cos(math.pi * current_steps / decay_step)) for base_lr in self.base_lrs]


class ConstantLRwithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        warmup_lr: float,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_steps:
            return [
                (base_lr - self.warmup_lr) * (self.last_epoch + 1) / self.warmup_steps + self.warmup_lr
                for base_lr in self.base_lrs
            ]
        else:
            return self.base_lrs

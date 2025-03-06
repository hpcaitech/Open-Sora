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

from ...apps.utils.dist import sync_tensor

__all__ = ["AverageMeter"]


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, is_distributed=True):
        self.is_distributed = is_distributed
        self.sum = 0
        self.count = 0

    def _sync(self, val: Union[torch.Tensor, int, float]) -> Union[torch.Tensor, int, float]:
        return sync_tensor(val, reduce="sum") if self.is_distributed else val

    def update(self, val: Union[torch.Tensor, int, float], delta_n=1):
        self.count += self._sync(delta_n)
        self.sum += self._sync(val * delta_n)

    def get_count(self) -> Union[torch.Tensor, int, float]:
        return self.count.item() if isinstance(self.count, torch.Tensor) and self.count.numel() == 1 else self.count

    @property
    def avg(self):
        avg = -1 if self.count == 0 else self.sum / self.count
        return avg.item() if isinstance(avg, torch.Tensor) and avg.numel() == 1 else avg

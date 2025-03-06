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

from typing import Any, Optional, Union

import numpy as np
import torch

__all__ = [
    "torch_randint",
    "torch_random",
    "torch_shuffle",
    "torch_uniform",
    "torch_random_choices",
]


def torch_randint(low: int, high: int, generator: Optional[torch.Generator] = None) -> int:
    """uniform: [low, high)"""
    if low == high:
        return low
    else:
        assert low < high
        return int(torch.randint(low=low, high=high, generator=generator, size=(1,)))


def torch_random(generator: Optional[torch.Generator] = None) -> float:
    """uniform distribution on the interval [0, 1)"""
    return float(torch.rand(1, generator=generator))


def torch_shuffle(src_list: list[Any], generator: Optional[torch.Generator] = None) -> list[Any]:
    rand_indexes = torch.randperm(len(src_list), generator=generator).tolist()
    return [src_list[i] for i in rand_indexes]


def torch_uniform(low: float, high: float, generator: Optional[torch.Generator] = None) -> float:
    """uniform distribution on the interval [low, high)"""
    rand_val = torch_random(generator)
    return (high - low) * rand_val + low


def torch_random_choices(
    src_list: list[Any],
    generator: Optional[torch.Generator] = None,
    k=1,
    weight_list: Optional[list[float]] = None,
) -> Union[Any, list]:
    if weight_list is None:
        rand_idx = torch.randint(low=0, high=len(src_list), generator=generator, size=(k,))
        out_list = [src_list[i] for i in rand_idx]
    else:
        assert len(weight_list) == len(src_list)
        accumulate_weight_list = np.cumsum(weight_list)

        out_list = []
        for _ in range(k):
            val = torch_uniform(0, accumulate_weight_list[-1], generator)
            active_id = 0
            for i, weight_val in enumerate(accumulate_weight_list):
                active_id = i
                if weight_val > val:
                    break
            out_list.append(src_list[active_id])

    return out_list[0] if k == 1 else out_list

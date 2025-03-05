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

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from ...models.nn.triton_rms_norm import TritonRMSNorm2dFunc
from ...models.utils import build_kwargs_from_config

__all__ = ["LayerNorm2d", "TritonRMSNorm2d", "build_norm", "reset_bn", "set_norm_eps"]


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


class TritonRMSNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TritonRMSNorm2dFunc.apply(x, self.weight, self.bias, self.eps)


class RMSNorm2d(nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = torch.nn.parameter.Parameter(torch.empty(self.num_features))
            if bias:
                self.bias = torch.nn.parameter.Parameter(torch.empty(self.num_features))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x / torch.sqrt(torch.square(x.float()).mean(dim=1, keepdim=True) + self.eps)).to(x.dtype)
        if self.elementwise_affine:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


# register normalization function here
REGISTERED_NORM_DICT: dict[str, type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
    "trms2d": TritonRMSNorm2d,
    "rms2d": RMSNorm2d,
}


def build_norm(name="bn2d", num_features=None, **kwargs) -> Optional[nn.Module]:
    if name in ["ln", "ln2d", "trms2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None


def reset_bn(
    model: nn.Module,
    data_loader: list,
    sync=True,
    progress_bar=False,
) -> None:
    import copy

    import torch.nn.functional as F
    from efficientvit.apps.utils import AverageMeter, is_master, sync_tensor
    from efficientvit.models.utils import get_device, list_join
    from tqdm import tqdm

    bn_mean = {}
    bn_var = {}

    tmp_model = copy.deepcopy(model)
    for name, m in tmp_model.named_modules():
        if isinstance(m, _BatchNorm):
            bn_mean[name] = AverageMeter(is_distributed=False)
            bn_var[name] = AverageMeter(is_distributed=False)

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    x = x.contiguous()
                    if sync:
                        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                        batch_mean = sync_tensor(batch_mean, reduce="cat")
                        batch_mean = torch.mean(batch_mean, dim=0, keepdim=True)

                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
                        batch_var = sync_tensor(batch_var, reduce="cat")
                        batch_var = torch.mean(batch_var, dim=0, keepdim=True)
                    else:
                        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.shape[0]
                    return F.batch_norm(
                        x,
                        batch_mean,
                        batch_var,
                        bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim],
                        False,
                        0.0,
                        bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    # skip if there is no batch normalization layers in the network
    if len(bn_mean) == 0:
        return

    tmp_model.eval()
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="reset bn", disable=not progress_bar or not is_master()) as t:
            for images in data_loader:
                images = images.to(get_device(tmp_model))
                tmp_model(images)
                t.set_postfix(
                    {
                        "bs": images.size(0),
                        "res": list_join(images.shape[-2:], "x"),
                    }
                )
                t.update()

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, _BatchNorm)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)


def set_norm_eps(model: nn.Module, eps: Optional[float] = None) -> None:
    for m in model.modules():
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm, _BatchNorm)):
            if eps is not None:
                m.eps = eps

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

import io
import os
from typing import Any

import onnx
import torch
import torch.nn as nn
from onnxsim import simplify as simplify_func

__all__ = ["export_onnx"]


def export_onnx(model: nn.Module, export_path: str, sample_inputs: Any, simplify=True, opset=11) -> None:
    """Export a model to a platform-specific onnx format.

    Args:
        model: a torch.nn.Module object.
        export_path: export location.
        sample_inputs: Any.
        simplify: a flag to turn on onnx-simplifier
        opset: int
    """
    model.eval()

    buffer = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(model, sample_inputs, buffer, opset_version=opset)
        buffer.seek(0, 0)
        if simplify:
            onnx_model = onnx.load_model(buffer)
            onnx_model, success = simplify_func(onnx_model)
            assert success
            new_buffer = io.BytesIO()
            onnx.save(onnx_model, new_buffer)
            buffer = new_buffer
            buffer.seek(0, 0)

    if buffer.getbuffer().nbytes > 0:
        save_dir = os.path.dirname(export_path)
        os.makedirs(save_dir, exist_ok=True)
        with open(export_path, "wb") as f:
            f.write(buffer.read())

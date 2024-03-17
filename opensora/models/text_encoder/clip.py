# Copyright 2024 Vchitect/Latte
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
# limitations under the License.# Modified from Latte
#
# This file is adapted from the Latte project.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Latte: https://github.com/Vchitect/Latte
# DiT:   https://github.com/facebookresearch/DiT/tree/main
# --------------------------------------------------------


import torch
import torch.nn as nn
import transformers
from transformers import CLIPTextModel, CLIPTokenizer

from opensora.registry import MODELS

transformers.logging.set_verbosity_error()


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, path="openai/clip-vit-huge-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(path)
        self.transformer = CLIPTextModel.from_pretrained(path)
        self.device = device
        self.max_length = max_length
        self._freeze()

    def _freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        pooled_z = outputs.pooler_output
        return z, pooled_z

    def encode(self, text):
        return self(text)


@MODELS.register_module("clip")
class ClipEncoder:
    """
    Embeds text prompt into vector representations. Also handles text dropout for classifier-free guidance.
    """

    def __init__(
        self,
        from_pretrained,
        model_max_length=77,
        device="cuda",
        dtype=torch.float,
    ):
        super().__init__()
        assert from_pretrained is not None, "Please specify the path to the T5 model"

        self.text_encoder = FrozenCLIPEmbedder(path=from_pretrained, max_length=model_max_length).to(device, dtype)
        self.y_embedder = None

        self.model_max_length = model_max_length
        self.output_dim = self.text_encoder.transformer.config.hidden_size

    def encode(self, text):
        _, pooled_embeddings = self.text_encoder.encode(text)
        y = pooled_embeddings.unsqueeze(1).unsqueeze(1)
        return dict(y=y)

    def null(self, n):
        null_y = self.y_embedder.y_embedding[None].repeat(n, 1, 1)[:, None]
        return null_y

    def to(self, dtype):
        self.text_encoder = self.text_encoder.to(dtype)
        return self

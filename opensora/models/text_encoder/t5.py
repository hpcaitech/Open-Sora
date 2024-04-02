# Adapted from PixArt
#
# Copyright (C) 2023  PixArt-alpha/PixArt-alpha
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# PixArt: https://github.com/PixArt-alpha/PixArt-alpha
# T5:     https://github.com/google-research/text-to-text-transfer-transformer
# --------------------------------------------------------


import torch
from transformers import AutoTokenizer, T5EncoderModel

from opensora.registry import MODELS


class T5Embedder:
    available_models = ["DeepFloyd/t5-v1_1-xxl"]

    def __init__(
        self,
        device,
        from_pretrained=None,
        *,
        cache_dir=None,
        hf_token=None,
        use_text_preprocessing=True,
        t5_model_kwargs=None,
        torch_dtype=None,
        use_offload_folder=None,
        model_max_length=120,
    ):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.cache_dir = cache_dir

        if t5_model_kwargs is None:
            t5_model_kwargs = {
                "low_cpu_mem_usage": True,
                "torch_dtype": self.torch_dtype,
            }

            if use_offload_folder is not None:
                t5_model_kwargs["offload_folder"] = use_offload_folder
                t5_model_kwargs["device_map"] = {
                    "shared": self.device,
                    "encoder.embed_tokens": self.device,
                    "encoder.block.0": self.device,
                    "encoder.block.1": self.device,
                    "encoder.block.2": self.device,
                    "encoder.block.3": self.device,
                    "encoder.block.4": self.device,
                    "encoder.block.5": self.device,
                    "encoder.block.6": self.device,
                    "encoder.block.7": self.device,
                    "encoder.block.8": self.device,
                    "encoder.block.9": self.device,
                    "encoder.block.10": self.device,
                    "encoder.block.11": self.device,
                    "encoder.block.12": "disk",
                    "encoder.block.13": "disk",
                    "encoder.block.14": "disk",
                    "encoder.block.15": "disk",
                    "encoder.block.16": "disk",
                    "encoder.block.17": "disk",
                    "encoder.block.18": "disk",
                    "encoder.block.19": "disk",
                    "encoder.block.20": "disk",
                    "encoder.block.21": "disk",
                    "encoder.block.22": "disk",
                    "encoder.block.23": "disk",
                    "encoder.final_layer_norm": "disk",
                    "encoder.dropout": "disk",
                }
            else:
                t5_model_kwargs["device_map"] = {
                    "shared": self.device,
                    "encoder": self.device,
                }

        self.use_text_preprocessing = use_text_preprocessing
        self.hf_token = hf_token

        assert from_pretrained in self.available_models
        self.tokenizer = AutoTokenizer.from_pretrained(
            from_pretrained, cache_dir=cache_dir
        )
        self.model = T5EncoderModel.from_pretrained(
            from_pretrained, cache_dir=cache_dir, **t5_model_kwargs
        ).eval()
        self.model_max_length = model_max_length

    def get_text_embeddings(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = text_tokens_and_mask["input_ids"].to(self.device)
        attention_mask = text_tokens_and_mask["attention_mask"].to(self.device)
        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )["last_hidden_state"].detach()
        return text_encoder_embs, attention_mask


@MODELS.register_module("t5")
class T5Encoder:
    def __init__(
        self,
        from_pretrained=None,
        model_max_length=120,
        device="cuda",
        dtype=torch.float,
        cache_dir=None,
        shardformer=False,
    ):
        assert from_pretrained is not None, "Please specify the path to the T5 model"

        self.t5 = T5Embedder(
            device=device,
            torch_dtype=dtype,
            from_pretrained=from_pretrained,
            cache_dir=cache_dir,
            model_max_length=model_max_length,
        )
        self.t5.model.to(dtype=dtype)
        self.y_embedder = None

        self.model_max_length = model_max_length
        self.output_dim = self.t5.model.config.d_model

        if shardformer:
            self.shardformer_t5()

    def shardformer_t5(self):
        from colossalai.shardformer import ShardConfig, ShardFormer

        from opensora.acceleration.shardformer.policy.t5_encoder import T5EncoderPolicy
        from opensora.utils.misc import requires_grad

        shard_config = ShardConfig(
            tensor_parallel_process_group=None,
            pipeline_stage_manager=None,
            enable_tensor_parallelism=False,
            enable_fused_normalization=False,
            enable_flash_attention=False,
            enable_jit_fused=True,
            enable_sequence_parallelism=False,
            enable_sequence_overlap=False,
        )
        shard_former = ShardFormer(shard_config=shard_config)
        optim_model, _ = shard_former.optimize(self.t5.model, policy=T5EncoderPolicy())
        self.t5.model = optim_model.half()

        # ensure the weights are frozen
        requires_grad(self.t5.model, False)

    def encode(self, text):
        caption_embs, emb_masks = self.t5.get_text_embeddings(text)
        caption_embs = caption_embs[:, None]
        return dict(y=caption_embs, mask=emb_masks)

    def null(self, n):
        null_y = self.y_embedder.y_embedding[None].repeat(n, 1, 1)[:, None]
        return null_y

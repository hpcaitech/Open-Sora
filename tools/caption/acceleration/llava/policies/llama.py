from typing import Dict, Union

import torch.nn as nn
from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["LlavaLlamaPolicy", "LlavaLlamaForCausalLMPolicy"]


class LlavaLlamaPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        if self.shard_config.enable_tensor_parallelism:
            # Resize embedding
            self.model.config.vocab_size
            self.shard_config.tensor_parallel_size

            # if vocab_size % world_size != 0:
            #     new_vocab_size = vocab_size + world_size - vocab_size % world_size
            #     self.model.resize_token_embeddings(new_vocab_size)

        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        policy = {}

        if self.shard_config.enable_tensor_parallelism:
            decoder_attribute_replacement = {
                "self_attn.hidden_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "self_attn.num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
            }
            if getattr(self.model.config, "num_key_value_heads", False):
                decoder_attribute_replacement["self_attn.num_key_value_heads"] = (
                    self.model.config.num_key_value_heads // self.shard_config.tensor_parallel_size
                )

            policy[LlamaDecoderLayer] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.gate_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.up_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.down_proj",
                        target_module=Linear1D_Row,
                    ),
                ],
            )

        return policy

    def postprocess(self):
        return self.model


class LlavaLlamaForCausalLMPolicy(LlavaLlamaPolicy):
    def module_policy(self):
        from transformers import LlamaForCausalLM

        policy = super().module_policy()
        if self.shard_config.enable_tensor_parallelism:
            # add a new item for casual lm
            new_item = {
                LlamaForCausalLM: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head", target_module=Linear1D_Col, kwargs={"gather_output": True}
                        )
                    ],
                )
            }
            policy.update(new_item)
        return policy

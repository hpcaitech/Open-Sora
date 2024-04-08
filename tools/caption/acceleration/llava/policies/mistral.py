import warnings
from typing import Dict, Union

import torch.nn as nn
from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row, VocabParallelEmbedding1D
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["LlavaMistralPolicy", "LlavaMistralForCausalLMPolicy"]


class LlavaMistralPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        if self.shard_config.enable_tensor_parallelism:
            # Resize embedding
            vocab_size = self.model.config.vocab_size
            world_size = self.shard_config.tensor_parallel_size

            if vocab_size % world_size != 0:
                new_vocab_size = vocab_size + world_size - vocab_size % world_size
                self.model.resize_token_embeddings(new_vocab_size)

        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralModel

        policy = {}

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "Mistral doesn't support sequence parallelism now, will ignore the sequence parallelism flag."
            )

        if self.shard_config.enable_tensor_parallelism:
            decoder_attribute_replacement = {
                "self_attn.hidden_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "self_attn.num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
                "self_attn.num_key_value_heads": self.model.config.num_key_value_heads
                // self.shard_config.tensor_parallel_size,
            }

            policy[MistralDecoderLayer] = ModulePolicyDescription(
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

            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="embed_tokens",
                    target_module=VocabParallelEmbedding1D,
                ),
                policy=policy,
                target_key=MistralModel,
            )

        return policy

    def postprocess(self):
        return self.model


class LlavaMistralForCausalLMPolicy(LlavaMistralPolicy):
    def module_policy(self):
        from transformers import MistralForCausalLM

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for casual lm
            new_item = {
                MistralForCausalLM: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head", target_module=Linear1D_Col, kwargs=dict(gather_output=True)
                        )
                    ]
                )
            }
            policy.update(new_item)
        return policy

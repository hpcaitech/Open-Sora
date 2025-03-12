from functools import partial
from typing import Dict, Union

import torch.nn as nn
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

from opensora.models.vae.tensor_parallel import Conv3dTPCol, Conv3dTPRow, GroupNormTP

from .distributed import ContextParallelAttention, TPUpDecoderBlockCausal3D, prepare_parallel_attention_mask
from .vae import DecoderCausal3D, EncoderCausal3D


def gen_resnets_replacements(prefix: str, with_shortcut: bool = False):
    replacements = [
        SubModuleReplacementDescription(
            suffix=f"{prefix}.norm1",
            target_module=GroupNormTP,
        ),
        SubModuleReplacementDescription(
            suffix=f"{prefix}.conv1.conv",
            target_module=Conv3dTPRow,
            kwargs=dict(
                split_output=True,
            ),
        ),
        SubModuleReplacementDescription(
            suffix=f"{prefix}.norm2",
            target_module=GroupNormTP,
        ),
        SubModuleReplacementDescription(
            suffix=f"{prefix}.conv2.conv",
            target_module=Conv3dTPRow,
            kwargs=dict(
                split_output=True,
            ),
        ),
    ]
    if with_shortcut:
        replacements.append(
            SubModuleReplacementDescription(
                suffix=f"{prefix}.conv_shortcut.conv",
                target_module=Conv3dTPRow,
                kwargs=dict(
                    split_output=True,
                ),
            )
        )
    return replacements


class HunyuanVaePolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        policy = {}

        policy[EncoderCausal3D] = ModulePolicyDescription(
            sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="conv_in.conv",
                    target_module=Conv3dTPCol,
                ),
                *gen_resnets_replacements("down_blocks[0].resnets[0]"),
                *gen_resnets_replacements("down_blocks[0].resnets[1]"),
                SubModuleReplacementDescription(
                    suffix="down_blocks[0].downsamplers[0].conv.conv",
                    target_module=Conv3dTPRow,
                    kwargs=dict(
                        split_output=True,
                    ),
                ),
                *gen_resnets_replacements("down_blocks[1].resnets[0]", with_shortcut=True),
                *gen_resnets_replacements("down_blocks[1].resnets[1]"),
                SubModuleReplacementDescription(
                    suffix="down_blocks[1].downsamplers[0].conv.conv",
                    target_module=Conv3dTPRow,
                ),
                SubModuleReplacementDescription(
                    suffix="mid_block.attentions[0]",
                    target_module=ContextParallelAttention,
                ),
            ],
            attribute_replacement={
                "down_blocks[0].downsamplers[0].channels": self.model.encoder.down_blocks[0].downsamplers[0].channels
                // self.shard_config.tensor_parallel_size,
                "down_blocks[1].downsamplers[0].channels": self.model.encoder.down_blocks[1].downsamplers[0].channels
                // self.shard_config.tensor_parallel_size,
                # "mid_block.attentions[0].processor": MemEfficientRingAttnProcessor(
                #     self.shard_config.tensor_parallel_process_group
                # ),
            },
            method_replacement={
                "prepare_attention_mask": partial(
                    prepare_parallel_attention_mask, cp_group=self.shard_config.tensor_parallel_process_group
                ),
            },
        )

        policy[DecoderCausal3D] = ModulePolicyDescription(
            sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="up_blocks[1].upsamplers[0]",
                    target_module=TPUpDecoderBlockCausal3D,
                    kwargs=dict(
                        split_output=True,
                    ),
                ),
                *gen_resnets_replacements("up_blocks[2].resnets[0]", with_shortcut=True),
                *gen_resnets_replacements("up_blocks[2].resnets[1]"),
                *gen_resnets_replacements("up_blocks[2].resnets[2]"),
                SubModuleReplacementDescription(
                    suffix="up_blocks[2].upsamplers[0].conv.conv",
                    target_module=Conv3dTPRow,
                    kwargs=dict(
                        split_output=True,
                    ),
                ),
                *gen_resnets_replacements("up_blocks[3].resnets[0]", with_shortcut=True),
                *gen_resnets_replacements("up_blocks[3].resnets[1]"),
                *gen_resnets_replacements("up_blocks[3].resnets[2]"),
                SubModuleReplacementDescription(
                    suffix="conv_norm_out",
                    target_module=GroupNormTP,
                ),
                SubModuleReplacementDescription(
                    suffix="conv_out.conv",
                    target_module=Conv3dTPRow,
                ),
                SubModuleReplacementDescription(
                    suffix="mid_block.attentions[0]",
                    target_module=ContextParallelAttention,
                ),
            ],
            attribute_replacement={
                "up_blocks[2].upsamplers[0].channels": self.model.decoder.up_blocks[2].upsamplers[0].channels
                // self.shard_config.tensor_parallel_size,
                # "mid_block.attentions[0].processor": MemEfficientRingAttnProcessor(
                #     self.shard_config.tensor_parallel_process_group
                # ),
            },
            method_replacement={
                "prepare_attention_mask": partial(
                    prepare_parallel_attention_mask, cp_group=self.shard_config.tensor_parallel_process_group
                ),
            },
        )

        return policy

    def postprocess(self):
        return self.model

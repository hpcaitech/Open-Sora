from colossalai.shardformer.modeling.jit import get_jit_fused_dropout_add_func
from colossalai.shardformer.modeling.t5 import get_jit_fused_T5_layer_ff_forward, get_T5_layer_self_attention_forward
from colossalai.shardformer.policies.base_policy import Policy, SubModuleReplacementDescription


class T5EncoderPolicy(Policy):
    def config_sanity_check(self):
        assert not self.shard_config.enable_tensor_parallelism
        assert not self.shard_config.enable_flash_attention

    def preprocess(self):
        return self.model

    def module_policy(self):
        from transformers.models.t5.modeling_t5 import T5LayerFF, T5LayerSelfAttention, T5Stack

        policy = {}

        # check whether apex is installed
        try:
            from opensora.acceleration.shardformer.modeling.t5 import T5LayerNorm

            # recover hf from fused rms norm to T5 norm which is faster
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="layer_norm",
                    target_module=T5LayerNorm,
                ),
                policy=policy,
                target_key=T5LayerFF,
            )
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(suffix="layer_norm", target_module=T5LayerNorm),
                policy=policy,
                target_key=T5LayerSelfAttention,
            )
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(suffix="final_layer_norm", target_module=T5LayerNorm),
                policy=policy,
                target_key=T5Stack,
            )
        except (ImportError, ModuleNotFoundError):
            pass

        # use jit operator
        if self.shard_config.enable_jit_fused:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_T5_layer_ff_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=T5LayerFF,
            )
            self.append_or_create_method_replacement(
                description={
                    "forward": get_T5_layer_self_attention_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=T5LayerSelfAttention,
            )

        return policy

    def postprocess(self):
        return self.model

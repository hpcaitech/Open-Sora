from colossalai.shardformer import ShardConfig, ShardFormer
from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

from opensora.acceleration.shardformer.policy.t5_encoder import T5EncoderPolicy
from opensora.registry import MODELS


@MODELS.register_module("text_embedder")
class HFEmbedder(nn.Module):
    def __init__(self, from_pretrained: str, max_length: int, shardformer: bool = False, **hf_kwargs):
        super().__init__()
        self.is_clip = "openai" in from_pretrained
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(from_pretrained, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(from_pretrained, **hf_kwargs)
            assert not shardformer, "Shardformer is not supported for CLIP"
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
                from_pretrained, max_length=max_length, legacy=True
            )
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(from_pretrained, **hf_kwargs)
            if shardformer:
                self.hf_module = shardformer_t5(self.hf_module)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str], added_tokens: int = 0, seq_align: int = 1) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        seq_len = batch_encoding["input_ids"].shape[1]
        if (added_tokens + seq_len) % seq_align != 0:
            num_pad_tokens = seq_align - (added_tokens + seq_len) % seq_align
            batch_encoding["input_ids"] = nn.functional.pad(
                batch_encoding["input_ids"], (0, num_pad_tokens), value=self.tokenizer.pad_token_id
            )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]


def shardformer_t5(t5: T5EncoderModel) -> T5EncoderModel:
    """
    Shardformer for T5 model

    Args:
        t5: T5 model to be optimized

    Returns:
        optimized T5 model
    """
    dtype = t5.shared.weight.dtype
    shard_config = ShardConfig(
        enable_tensor_parallelism=False,
        enable_jit_fused=True,
    )
    shard_former = ShardFormer(shard_config=shard_config)
    optim_model, _ = shard_former.optimize(t5, policy=T5EncoderPolicy())
    optim_model = optim_model.to(dtype).eval().requires_grad_(False)
    return optim_model

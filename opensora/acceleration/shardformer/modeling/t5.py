import torch
import torch.nn as nn


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

    @staticmethod
    def from_native_module(module, *args, **kwargs):
        assert module.__class__.__name__ == "FusedRMSNorm", (
            "Recovering T5LayerNorm requires the original layer to be apex's Fused RMS Norm."
            "Apex's fused norm is automatically used by Hugging Face Transformers https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L265C5-L265C48"
        )

        layer_norm = T5LayerNorm(module.normalized_shape, eps=module.eps)
        layer_norm.weight.data.copy_(module.weight.data)
        layer_norm = layer_norm.to(module.weight.device)
        return layer_norm

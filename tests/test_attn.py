import torch
from colossalai.accelerator import get_accelerator
from colossalai.utils import get_current_device
from rotary_embedding_torch import RotaryEmbedding

from opensora.models.layers.blocks import Attention

# B, S, H = 7488, 1, 1152
# B, S, H = 32, 234, 1152
B, S, H = 128, 32, 1152
N, D = 16, 72


def run_attn(enable_flash_attn: bool):
    get_accelerator().reset_peak_memory_stats()
    rope = RotaryEmbedding(D).to(device=get_current_device(), dtype=torch.bfloat16)
    attn = Attention(
        H,
        N,
        qkv_bias=True,
        rope=rope.rotate_queries_or_keys,
        enable_flash_attn=enable_flash_attn,
    ).to(device=get_current_device(), dtype=torch.bfloat16)
    x = torch.randn(B, S, H, device=get_current_device(), dtype=torch.bfloat16).requires_grad_()
    y = attn(x)
    y.mean().backward()
    print(f"Peak memory: {get_accelerator().max_memory_allocated() / 1024**2:.2f} MB")


if __name__ == "__main__":
    print("Use flashattn")
    run_attn(True)
    print("No flashattn")
    run_attn(False)

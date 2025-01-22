from itertools import product

import pytest
import torch
from colossalai.accelerator import get_accelerator
from colossalai.utils import get_current_device

from opensora.models.layers.blocks import Attention, split_batch_cat_seq, split_seq_cat_batch
from opensora.models.layers.rotary_embedding_torch import RotaryEmbedding

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


def test_block_transform():
    b, h, w, c = 8, 12, 4, 3
    x = torch.randn(b, h, w, c)
    kernel_sizes = (3, 2)
    dims = (1, 2)
    num_splits = [x.size(d) // k for d, k in zip(dims, kernel_sizes)]
    y = split_seq_cat_batch(x, kernel_sizes, dims)
    z = split_batch_cat_seq(y, b, num_splits, dims)
    assert torch.equal(x, z)


@pytest.mark.parametrize(
    "shape, kernel_sizes",
    [
        [(8, 12, 4, 1), (2, 2, -1)],  # divisible + N<B + 3D
        [(1, 5, 2, 5), (2, 2, -1)],  # undivisible + N>B + 3D
    ],
)
@pytest.mark.parametrize("shift_window", [False, True])
def test_block_attn_nd(shape, kernel_sizes, shift_window):
    hidden_size = 96
    num_heads = 4
    head_dim = hidden_size // num_heads
    rope = RotaryEmbedding(head_dim // 3).to(device=get_current_device(), dtype=torch.bfloat16)
    attn = Attention(
        hidden_size,
        num_heads,
        qkv_bias=True,
        qk_norm=True,
        enable_flash_attn=True,
        rope=rope.rotate_queries_or_keys,
        kernel_size=kernel_sizes,
        shift_window=shift_window,
    ).to(device=get_current_device(), dtype=torch.bfloat16)
    # [B, H, W, C]
    x = torch.rand(*shape, hidden_size, device=get_current_device(), dtype=torch.bfloat16).requires_grad_()
    y = attn(x)
    assert x.shape == y.shape
    loss = y.mean()
    loss.backward()


@pytest.mark.parametrize(
    "shape, kernel_sizes",
    [
        [(8, 12, 4, 1), (2, 2, -1)],  # divisible + N<B + 3D
        [(8, 12, 4, 6), (2, 2, -1)],  # divisible + N<B + 3D
        [(8, 12, 3, 6), (2, 2, -1)],  # divisible + N<B + 3D
        [(8, 90, 60, 13), (8, 8, -1)],  # 480p video
        [(8, 160, 90, 1), (8, 8, -1)],  # 720p image
    ],
)
def test_block_attn_3d(shape, kernel_sizes):
    hidden_size = 96
    num_heads = 4
    head_dim = hidden_size // num_heads
    rope = RotaryEmbedding(head_dim // 3).to(device=get_current_device(), dtype=torch.bfloat16)
    attn = Attention(
        hidden_size,
        num_heads,
        qkv_bias=True,
        qk_norm=True,
        enable_flash_attn=False,
        rope=rope.rotate_queries_or_keys,
        kernel_size=kernel_sizes,
    ).to(device=get_current_device(), dtype=torch.bfloat16)
    # [B, H, W, T, C]
    x = torch.rand(*shape, hidden_size, device=get_current_device(), dtype=torch.bfloat16)
    y = attn(x)

    split_size = [k if k > 0 else x.size(i + 1) for i, k in enumerate(kernel_sizes)]
    for start_indices in product(*[range(0, x.size(i + 1), s) for i, s in enumerate(split_size)]):
        piece = x[
            :,
            start_indices[0] : start_indices[0] + split_size[0],
            start_indices[1] : start_indices[1] + split_size[1],
            start_indices[2] : start_indices[2] + split_size[2],
            :,
        ]
        piece_z = attn(piece)
        piece_y = y[
            :,
            start_indices[0] : start_indices[0] + split_size[0],
            start_indices[1] : start_indices[1] + split_size[1],
            start_indices[2] : start_indices[2] + split_size[2],
            :,
        ]
        assert piece_y.shape == piece_z.shape
        assert torch.equal(
            piece_z,
            piece_y,
        )


@pytest.mark.parametrize(
    "shape, kernel_sizes, kernel_sizes2",
    [
        [(2, 80, 60, 1), (8, 8, 4), (8, 8, -1)],  # 720p image
        [(2, 4, 4, 6), (4, 4, -1), (8, 8, -1)],  # 720p image
    ],
)
def test_block_attn_3d_var_kernel(shape, kernel_sizes, kernel_sizes2):
    hidden_size = 24
    num_heads = 2
    head_dim = hidden_size // num_heads
    rope = RotaryEmbedding(head_dim // 3).to(device=get_current_device(), dtype=torch.bfloat16)
    attn = Attention(
        hidden_size,
        num_heads,
        qkv_bias=True,
        qk_norm=True,
        enable_flash_attn=False,
        rope=rope.rotate_queries_or_keys,
        kernel_size=kernel_sizes,
    ).to(device=get_current_device(), dtype=torch.bfloat16)
    # [B, H, W, T, C]
    x = torch.rand(*shape, hidden_size, device=get_current_device(), dtype=torch.bfloat16)
    y = attn(x)
    attn2 = Attention(
        hidden_size,
        num_heads,
        qkv_bias=True,
        qk_norm=True,
        enable_flash_attn=False,
        rope=rope.rotate_queries_or_keys,
        kernel_size=kernel_sizes2,
    ).to(device=get_current_device(), dtype=torch.bfloat16)
    attn2.load_state_dict(attn.state_dict())
    y2 = attn2(x)
    assert y.shape == y2.shape
    torch.testing.assert_close(y, y2)


def test_block_attn_3d_overlap():
    kernel_sizes = (8, 8, -1)
    hidden_size = 24
    num_heads = 2
    head_dim = hidden_size // num_heads
    rope = RotaryEmbedding(head_dim // 3).to(device=get_current_device(), dtype=torch.bfloat16)
    attn = Attention(
        hidden_size,
        num_heads,
        qkv_bias=True,
        qk_norm=True,
        enable_flash_attn=False,
        rope=rope.rotate_queries_or_keys,
        kernel_size=kernel_sizes,
    ).to(device=get_current_device(), dtype=torch.bfloat16)
    # [B, H, W, T, C]
    x = torch.rand(2, 40, 40, 6, hidden_size, device=get_current_device(), dtype=torch.bfloat16)
    y = attn(x)
    x2 = torch.rand(2, 48, 48, 6, hidden_size, device=get_current_device(), dtype=torch.bfloat16)
    x2[:, :40, :40] = x
    y2 = attn(x2)
    torch.testing.assert_close(y, y2[:, :40, :40])


def test_block_transform_3d():
    b, h, w, t, c = 8, 12, 4, 6, 3
    x = torch.randn(b, h, w, t, c)
    kernel_sizes = (3, 2, 6)
    dims = (1, 2, 3)
    num_splits = [x.size(d) // k for d, k in zip(dims, kernel_sizes)]
    y = split_seq_cat_batch(x, kernel_sizes, dims)
    split_size = [k if k > 0 else x.size(i + 1) for i, k in enumerate(kernel_sizes)]
    for i, start_indices in enumerate(product(*[range(0, x.size(i + 1), s) for i, s in enumerate(split_size)])):
        piece = x[
            :,
            start_indices[0] : start_indices[0] + split_size[0],
            start_indices[1] : start_indices[1] + split_size[1],
            start_indices[2] : start_indices[2] + split_size[2],
            :,
        ]
        y_piece = y[i * b : (i + 1) * b]
        assert torch.equal(y_piece, piece), f"{y_piece.shape} vs {piece.shape}"

    z = split_batch_cat_seq(y, b, num_splits, dims)
    assert torch.equal(x, z)


if __name__ == "__main__":
    print("Use flashattn")
    run_attn(True)
    print("No flashattn")
    run_attn(False)

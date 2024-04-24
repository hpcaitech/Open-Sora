import pytest
import torch

from opensora.models.layers.blocks import PositionEmbedding2D, get_2d_sincos_pos_embed

D = 8
SCALE = 2.0
from torch.testing import assert_close


def get_spatial_pos_embed(x, hidden_size, h, w, scale, base_size=None):
    pos_embed = get_2d_sincos_pos_embed(
        hidden_size,
        (h, w),
        scale=scale,
        base_size=base_size,
    )
    pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
    return pos_embed.to(device=x.device, dtype=x.dtype)


@pytest.mark.parametrize("dtype", [torch.float, torch.float16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pos_emb(dtype, device):
    # just a placeholder to get the device and dtype
    x = torch.empty(1, dtype=dtype, device=device)
    pos_embedder = PositionEmbedding2D(
        D,
        max_position_embeddings=8,
        scale=SCALE,
    ).to(device=device, dtype=dtype)
    output = pos_embedder(x, 8, 7)
    target = get_spatial_pos_embed(x, D, 8, 7, SCALE)
    assert_close(output, target)
    output = pos_embedder(x, 15, 16)
    target = get_spatial_pos_embed(x, D, 15, 16, SCALE)
    assert_close(output, target)
    output = pos_embedder(x, 30, 20, base_size=2)
    target = get_spatial_pos_embed(x, D, 30, 20, SCALE, base_size=2)
    assert_close(output, target)
    # test cache
    output = pos_embedder(x, 30, 20, base_size=2)
    target = get_spatial_pos_embed(x, D, 30, 20, SCALE, base_size=2)
    assert_close(output, target)
    assert pos_embedder._get_cached_emb.cache_info().hits >= 1

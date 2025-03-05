import torch
from einops import rearrange
from flash_attn import flash_attn_func as flash_attn_func_v2
from liger_kernel.ops.rope import LigerRopeFunction
from torch import Tensor, Tuple

try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3

    SUPPORT_FA3 = True
except:
    SUPPORT_FA3 = False


def flash_attn_func(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    if SUPPORT_FA3:
        return flash_attn_func_v3(q, k, v)[0]
    return flash_attn_func_v2(q, k, v)


def attention(q: Tensor, k: Tensor, v: Tensor, pe) -> Tensor:
    if isinstance(pe, torch.Tensor):
        q, k = apply_rope(q, k, pe)
    else:
        cos, sin = pe
        q, k = LigerRopeFunction.apply(q, k, cos, sin)
        # to compare with the original implementation
        # k = reverse_rearrange_tensor(k)
    q = rearrange(q, "B H L D -> B L H D")
    k = rearrange(k, "B H L D -> B L H D")
    v = rearrange(v, "B H L D -> B L H D")
    x = flash_attn_func(q, k, v)
    x = rearrange(x, "B L H D -> B L (H D)")

    return x


def liger_rope(pos: Tensor, dim: int, theta: int) -> Tuple:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)  # (b, seq, dim//2)
    cos = out.cos()
    sin = out.sin()

    return (cos, sin)


def rope(pos: Tensor, dim: int, theta: int) -> Tuple:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def rearrange_tensor(tensor):
    """
    Rearranges the last dimension (D) of the input tensor based on the specified mapping:
    2d -> d, 2d+1 -> D/2 + d.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, H, L, D], where D is even.

    Returns:
        torch.Tensor: Tensor with rearranged last dimension, same shape as input.
    """
    B, H, L, D = tensor.shape
    if D % 2 != 0:
        raise ValueError("The last dimension D must be even.")

    half_D = D // 2
    indices = torch.empty(D, dtype=torch.long, device=tensor.device)

    # Fill the indices based on the mapping rule
    indices[:half_D] = torch.arange(0, D, 2, device=tensor.device)
    indices[half_D:] = torch.arange(1, D, 2, device=tensor.device)

    # Rearrange the tensor based on the computed indices
    return tensor.index_select(dim=-1, index=indices)


def reverse_rearrange_tensor(tensor):
    """
    Restores the original order of the last dimension (D) of the input tensor based on the reverse mapping:
    d -> 2d, D/2 + d -> 2d + 1.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, H, L, D], where D is even.

    Returns:
        torch.Tensor: Tensor with restored original last dimension order, same shape as input.
    """
    B, H, L, D = tensor.shape
    if D % 2 != 0:
        raise ValueError("The last dimension D must be even.")

    half_D = D // 2
    reverse_indices = torch.empty(D, dtype=torch.long, device=tensor.device)

    # Fill the reverse indices to restore the original order
    reverse_indices[::2] = torch.arange(half_D, device=tensor.device)
    reverse_indices[1::2] = torch.arange(half_D, D, device=tensor.device)

    # Rearrange the tensor based on the reverse indices
    return tensor.index_select(dim=-1, index=reverse_indices)

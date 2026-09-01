import importlib.util
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

UTILS_PATH = Path(__file__).parents[1] / "opensora" / "models" / "vae" / "utils.py"
SPEC = importlib.util.spec_from_file_location("opensora_vae_utils", UTILS_PATH)
vae_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vae_utils)

ChannelChunkConv3d = vae_utils.ChannelChunkConv3d


@pytest.mark.parametrize(
    ("in_channels", "out_channels", "groups", "bias", "batched"),
    [
        pytest.param(4, 6, 1, False, True, id="dense-no-bias"),
        pytest.param(4, 6, 2, True, True, id="grouped"),
        pytest.param(4, 8, 4, False, True, id="depthwise-multiplier-no-bias"),
        pytest.param(3, 5, 1, True, False, id="unbatched"),
    ],
)
def test_chunked_forward_and_gradients_match_native_conv3d(
    monkeypatch, in_channels, out_channels, groups, bias, batched
):
    monkeypatch.setattr(ChannelChunkConv3d, "CONV3D_NUMEL_LIMIT", 128)
    torch.manual_seed(2026)
    conv = ChannelChunkConv3d(
        in_channels,
        out_channels,
        kernel_size=(2, 2, 2),
        stride=(1, 1, 2),
        padding=(1, 0, 1),
        groups=groups,
        bias=bias,
        dtype=torch.double,
    )

    input_shape = ((1,) if batched else ()) + (in_channels, 3, 4, 5)
    actual_input = torch.randn(input_shape, dtype=torch.double, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_()
    reference_weight = conv.weight.detach().clone().requires_grad_()
    reference_bias = conv.bias.detach().clone().requires_grad_() if bias else None

    actual = conv(actual_input)
    expected = F.conv3d(
        reference_input,
        reference_weight,
        reference_bias,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    output_grad = torch.randn_like(expected)
    actual_parameters = (actual_input, conv.weight) + ((conv.bias,) if bias else ())
    reference_parameters = (reference_input, reference_weight) + (
        (reference_bias,) if bias else ()
    )
    actual_grads = torch.autograd.grad(actual, actual_parameters, output_grad)
    reference_grads = torch.autograd.grad(expected, reference_parameters, output_grad)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-12, atol=1e-12)


def test_output_size_alone_activates_chunked_path(monkeypatch):
    monkeypatch.setattr(ChannelChunkConv3d, "CONV3D_NUMEL_LIMIT", 64)
    torch.manual_seed(7)
    conv = ChannelChunkConv3d(1, 16, kernel_size=1, dtype=torch.double)
    input_tensor = torch.randn(1, 1, 2, 2, 2, dtype=torch.double)
    expected = F.conv3d(input_tensor, conv.weight, conv.bias)

    native_conv3d = F.conv3d
    with mock.patch.object(vae_utils.F, "conv3d", wraps=native_conv3d) as conv3d_spy:
        actual = conv(input_tensor)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert conv3d_spy.call_count > 1


def test_chunked_path_preserves_autocast_output_dtype(monkeypatch):
    monkeypatch.setattr(ChannelChunkConv3d, "CONV3D_NUMEL_LIMIT", 100)
    torch.manual_seed(17)
    conv = ChannelChunkConv3d(4, 6, kernel_size=1)
    input_tensor = torch.randn(1, 4, 3, 4, 5)

    native_conv3d = F.conv3d
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = F.conv3d(input_tensor, conv.weight, conv.bias)
        with mock.patch.object(
            vae_utils.F, "conv3d", wraps=native_conv3d
        ) as conv3d_spy:
            actual = conv(input_tensor)

    assert actual.dtype == expected.dtype == torch.bfloat16
    assert conv3d_spy.call_count > 1
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_functional_path_preserves_group_connectivity():
    torch.manual_seed(99)
    input_tensor = torch.randn(1, 4, 3, 4, 5, dtype=torch.double)
    weight = torch.randn(6, 2, 2, 2, 2, dtype=torch.double)
    bias = torch.randn(6, dtype=torch.double)
    conv_args = ((1, 1, 2), (1, 0, 1), (1, 1, 1), 2)

    expected = F.conv3d(input_tensor, weight, bias, *conv_args)
    actual = vae_utils.channel_chunk_conv3d(
        input_tensor, weight, bias, *conv_args, numel_limit=128
    )

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("padding", ["same", "valid"])
def test_string_padding_matches_native_conv3d(monkeypatch, padding):
    monkeypatch.setattr(ChannelChunkConv3d, "CONV3D_NUMEL_LIMIT", 32)
    torch.manual_seed(123)
    conv = ChannelChunkConv3d(2, 3, kernel_size=3, padding=padding, dtype=torch.double)
    input_tensor = torch.randn(1, 2, 4, 5, 6, dtype=torch.double)

    expected = F.conv3d(input_tensor, conv.weight, conv.bias, padding=padding)
    actual = conv(input_tensor)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_nonzero_padding_mode_matches_native_conv3d(monkeypatch):
    monkeypatch.setattr(ChannelChunkConv3d, "CONV3D_NUMEL_LIMIT", 32)
    torch.manual_seed(456)
    conv = ChannelChunkConv3d(
        2, 3, kernel_size=3, padding=1, padding_mode="reflect", dtype=torch.double
    )
    input_tensor = torch.randn(1, 2, 4, 5, 6, dtype=torch.double)

    expected = nn.Conv3d.forward(conv, input_tensor)
    actual = conv(input_tensor)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

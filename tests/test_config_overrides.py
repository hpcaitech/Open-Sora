import sys
from pathlib import Path

import pytest
from mmengine.config import Config

from opensora.utils.config import parse_configs

INFERENCE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs/diffusion/inference/256px.py"
)


def parse_cli(monkeypatch, *args, config_path=INFERENCE_CONFIG):
    monkeypatch.setattr(
        sys, "argv", ["scripts/diffusion/inference.py", str(config_path), *args]
    )
    return parse_configs()


@pytest.mark.parametrize(
    "flag,value,path,expected_type,expected",
    [
        ("--model.from-pretrained", "123", "model.from_pretrained", str, "123"),
        ("--sampling_option.guidance", "2", "sampling_option.guidance", float, 2.0),
        ("--sampling_option.seed", "42", "sampling_option.seed", int, 42),
        ("--model.axes_dim", "[8, 16, 16]", "model.axes_dim", list, [8, 16, 16]),
        ("--sampling_option.shift", "false", "sampling_option.shift", bool, False),
        ("--model.depth", "3", "model.depth", int, 3),
    ],
)
def test_published_inference_config_leaf_types(
    monkeypatch, flag, value, path, expected_type, expected
):
    cfg = parse_cli(monkeypatch, flag, value)
    actual = cfg
    for key in path.split("."):
        actual = actual[key]
    assert type(actual) is expected_type
    assert actual == expected
    assert cfg.config_path == str(INFERENCE_CONFIG)


def test_nested_native_config_path(monkeypatch, tmp_path):
    config_path = tmp_path / "nested.py"
    config_path.write_text("model = dict(block=dict(attention=dict(depth=4)))\n")
    cfg = parse_cli(
        monkeypatch, "--model.block.attention.depth", "8", config_path=config_path
    )
    assert cfg.model.block.attention.depth == 8


def test_literal_mapping_override(monkeypatch):
    cfg = parse_cli(monkeypatch, "--dataset", "{'type': 'text', 'path': 'prompts.csv'}")
    assert cfg.dataset == {"type": "text", "path": "prompts.csv"}


def test_new_leaf_and_none(monkeypatch):
    cfg = parse_cli(
        monkeypatch,
        "--sampling_option.new_flag",
        "True",
        "--model.from_pretrained",
        "none",
    )
    assert cfg.sampling_option.new_flag is True
    assert cfg.model.from_pretrained is None


def test_missing_parent_is_rejected(monkeypatch):
    with pytest.raises(AssertionError, match="Key missing not found in config"):
        parse_cli(monkeypatch, "--model.missing.depth", "8")


def test_top_level_scalar_types_are_preserved(monkeypatch):
    cfg = parse_cli(monkeypatch, "--motion-score", "5", "--seed", "7")
    assert cfg.motion_score == "5"
    assert type(cfg.motion_score) is str
    assert cfg.seed == 7
    assert type(cfg.seed) is int


def test_inherited_native_config(monkeypatch, tmp_path):
    config_path = tmp_path / "inherited.py"
    config_path.write_text(f"_base_ = {str(INFERENCE_CONFIG)!r}\n")
    cfg = parse_cli(
        monkeypatch, "--sampling_option.guidance", "2", config_path=config_path
    )
    assert type(cfg.sampling_option.guidance) is float
    assert cfg.sampling_option.guidance == 2.0
    assert Config.fromfile(config_path).sampling_option.guidance == 7.5

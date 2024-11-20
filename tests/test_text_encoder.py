# ClipEncoder
import torch
import torch.nn as nn
import torch_musa
from opensora.models.text_encoder.clip import FrozenCLIPEmbedder, ClipEncoder
from opensora.models.text_encoder.t5 import T5Encoder
from colossalai.testing import parameterize


def test_clip():
    device = torch.device("musa")
    torch.manual_seed(1024)
    pass


if __name__ == "__main__":
    test_clip()
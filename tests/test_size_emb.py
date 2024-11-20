import copy
import torch
import torch_musa
from torch.testing import assert_close
from opensora.models.layers.blocks import SizeEmbedder


def test_size_emb(device):
    B = 1
    s = torch.randn(B, 256, device=device)
    bs= B
    size_embedder = SizeEmbedder(
       hidden_size=256
    ).to(device=device)
    output = size_embedder(s, bs)
    output.sum().backward()
    print(f"Shape {output.shape}\n {output}\n")


def test_size_emb_correctness(device):
    B = 1
    s = torch.randn(B, 256)
    bs= B
    
    s_musa = copy.deepcopy(s).to(device=device)
    
    size_embedder = SizeEmbedder(
       hidden_size=256
    )
    size_embedder_musa = copy.deepcopy(size_embedder).to(device=device)
    
    
    output = size_embedder(s, bs)
    output_musa = size_embedder_musa(s_musa, bs)
    
    assert_close(output, output_musa, check_device=False)
    
    output.sum().backward()
    output_musa.sum().backward()


if __name__ == "__main__":
    test_size_emb("musa")
    test_size_emb_correctness("musa")
import copy
import torch
import torch_musa
from torch.testing import assert_close
from opensora.models.layers.blocks import CaptionEmbedder

def test_caption_emb(device):
    dtype = torch.bfloat16
    x = torch.randn(120, 4, device=device, dtype=dtype).requires_grad_()
    caption_embedder = CaptionEmbedder(
        in_channels=4,
        hidden_size=256,
        uncond_prob=0.5,
    ).to(device=device, dtype=dtype)
    output = caption_embedder(caption=x, train=False)
    output.sum().backward()
    print(f"Shape {output.shape}\n {output}\n")

# TODO: distributed test; may not;
# TODO: correctness test 
def test_caption_emb_correctness():
    dtype = torch.bfloat16
    device="musa"
    torch.manual_seed(1024)
    
    x_cpu = torch.randn(120, 4, dtype=dtype).requires_grad_()
    x_musa = copy.deepcopy(x_cpu).to(device=device)
    
    caption_embedder_cpu = CaptionEmbedder(
        in_channels=4,
        hidden_size=256,
        uncond_prob=0.5,
    ).to(dtype=dtype)
    caption_embedder_musa = copy.deepcopy(caption_embedder_cpu).to(device=device)
    
    output_cpu = caption_embedder_cpu(caption=x_cpu, train=False)
    output_musa = caption_embedder_musa(caption=x_musa, train=False)
    
    assert_close(output_cpu, output_musa, check_device=False)
        

if __name__ == "__main__":
    print("Test Caption Embedding")
    test_caption_emb("musa")
    
    print("Test Caption Embedding Correctness")
    test_caption_emb_correctness()
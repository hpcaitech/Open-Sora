import copy
import torch
import torch_musa
from torch.testing import assert_close
from opensora.models.layers.blocks import PatchEmbed3D

def test_patch_emb_3D(device):
    dtype = torch.float32
    x = torch.randn(4, 3, 32, 64, 64, device=device, dtype=dtype).requires_grad_()
    patch_embedder = PatchEmbed3D().to(device=device, dtype=dtype)
    output = patch_embedder(x)
    output.sum().backward()
    print(f"Shape {output.shape}\n {output}\n")

# TODO: distributed test; may not;
# TODO: correctness test 
def test_patch_emb_3D_correctness():
    dtype = torch.float32
    device="musa"
    torch.manual_seed(1024)
    
    x_cpu = torch.randn(4, 3, 32, 64, 64, dtype=dtype).requires_grad_()
    x_musa = copy.deepcopy(x_cpu).to(device=device)
    
    patch_embedder_cpu = PatchEmbed3D().to(dtype=dtype)
    patch_embedder_musa = copy.deepcopy(patch_embedder_cpu).to(device=device)
    
    output_cpu = patch_embedder_cpu(x_cpu)
    output_musa = patch_embedder_musa(x_musa)
    
    assert_close(output_cpu, output_musa, check_device=False)
    

if __name__ == "__main__":
    print("Test Patch Embedding 3D")
    test_patch_emb_3D("musa")
    
    print("Test Patch Embedding 3D Correctness")
    test_patch_emb_3D_correctness()
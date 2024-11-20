import copy
import torch
import torch_musa
from torch.testing import assert_close
from opensora.models.layers.blocks import TimestepEmbedder

def test_timestep_emb(device):
    dtype = torch.bfloat16 # torch.float, torch.float16, torch.bfloat16
    x = torch.empty(1, dtype=dtype, device=device)
    timestep_embedder = TimestepEmbedder(
       hidden_size=256
    ).to(device=device, dtype=dtype)
    output = timestep_embedder(x, dtype)
    print(f"Shape {output.shape}\n {output}\n")
    

# TODO: distributed test; may not;
# TODO: correctness test 
def test_timestep_emb_correctnes():
    dtype = torch.bfloat16 # torch.float, torch.float16, torch.bfloat16
    device="musa"
    torch.manual_seed(1024)
    
    x_cpu = torch.empty(1, dtype=dtype)
    x_musa = torch.empty(1, dtype=dtype, device=device)
    
    timestep_embedder_cpu = TimestepEmbedder(
       hidden_size=256
    ).to(dtype=dtype)
    timestep_embedder_musa = copy.deepcopy(timestep_embedder_cpu).to(device=device, dtype=dtype)
    
    output_cpu = timestep_embedder_cpu(x_cpu, dtype)
    output_musa = timestep_embedder_musa(x_musa, dtype)

    assert_close(output_cpu, output_musa, check_device=False)
    

if __name__ == "__main__":
    print("Test Timestep Embedder Embedding")
    test_timestep_emb("musa")
    
    print("Test Timestep Embedder Embedding Correctness")
    test_timestep_emb_correctnes()
    
    
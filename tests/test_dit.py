import copy
import torch
import torch.nn as nn
import torch_musa
import torch.distributed as dist
from torch.testing import assert_close
from opensora.models.dit.dit import DiTBlock
from opensora.models.dit import DiT, DiT_XL_2, DiT_XL_2x2
from opensora.acceleration.parallel_states import set_sequence_parallel_group

# ditblock run test
def test_ditblock(device):
    device = torch.device(device)
    torch.manual_seed(1024)
    dtype = torch.float32
    
    N, T, D = 4, 64, 256
    N_token = 256
    device = torch.device(device)
    
    dit_block = DiTBlock(hidden_size=256, num_heads=8).to(device)
    
    x = torch.randn(N, T, D, dtype=dtype).to(device)  # (N, T, D)
    x.requires_grad = True
    c = torch.randn(N, D, dtype=dtype).to(device)  #  [N, D]
    c.requires_grad = True
    
    output = dit_block(x,c)
    print(f"dit_block Shape {output.shape}\n {output}\n")
    
    output.mean().backward()


# ditblock correctness test
def test_ditblock_correctness(device):
    N, T, D = 4, 64, 256
    device = torch.device(device)
    torch.manual_seed(1024)
    dtype = torch.float32
    
    x_cpu = torch.randn(N, T, D ).to(dtype=dtype)
    x_cpu.requires_grad = True
    c_cpu = torch.randn(N, D).to(dtype=dtype)
    c_cpu.requires_grad = True
    
    x_musa = copy.deepcopy(x_cpu).to(device=device)
    c_musa = copy.deepcopy(c_cpu).to(device=device)
    
    dit_block_cpu = DiTBlock(hidden_size=256, num_heads=8)
    dit_block_musa = copy.deepcopy(dit_block_cpu).to(device=device)
    
    output_cpu = dit_block_cpu(x_cpu, c_cpu)
    output_musa = dit_block_musa(x_musa, c_musa)
    
    print(f"dit_block_cpu Shape {output_cpu.shape}\n {output_cpu}\n")
    print(f"dit_block_musa Shape {output_musa.shape}\n {output_musa}\n")
    
    output_cpu.mean().backward()
    output_musa.mean().backward()
    
    assert_close(output_cpu, output_musa, check_device=False)


# dit run test 
def test_dit(device):
    device = torch.device(device)
    torch.manual_seed(1024)
    dtype = torch.float32
    
    # N, T, D = 4, 64, 256
    B, C, T, H, W = 1, 4, 64, 16, 16
    # x: (B, C, T, H, W) tensor of inputs
    # t: (B,) tensor of diffusion timesteps
    # y: list of text
   
    dit = DiT().to(device)
    
    x = torch.randn(B, C, T, H, W, dtype=dtype).to(device)  # (B, C, T, H, W)
    x.requires_grad = True
    y = torch.randn(B, 1, 1, 512, dtype=dtype).to(device)   #  [B, caption_channels=512]
    timestep = torch.randn(B, dtype=dtype).to(device) #  [B, ]
   
    x_dit = dit(x, timestep, y)

    print(f"DiT Shape {x_dit.shape}\n {x_dit}\n")
    
    x_dit.mean().backward()


# ditblock correctness test
def test_dit_correctness(device):
    B, C, T, H, W = 1, 4, 64, 16, 16
    # x: (B, C, T, H, W) tensor of inputs
    # t: (B,) tensor of diffusion timesteps
    # y: list of text
    device = torch.device(device)
    torch.manual_seed(1024)
    dtype = torch.float32
    
    x_cpu = torch.randn(B, C, T, H, W, dtype=dtype)  # (B, C, T, H, W)
    x_cpu.requires_grad = True
    y_cpu = torch.randn(B, 1, 1, 512, dtype=dtype)   #  [B, caption_channels=512]
    timestep_cpu = torch.randn(B, dtype=dtype) #  [B,]
    
    x_musa = copy.deepcopy(x_cpu).to(device=device)
    y_musa = copy.deepcopy(y_cpu).to(device=device)
    timestep_musa = copy.deepcopy(timestep_cpu).to(device=device)
    
    dit_cpu = DiT()
    dit_musa = copy.deepcopy(dit_cpu).to(device=device)
    
    output_cpu = dit_cpu(x_cpu, timestep_cpu, y_cpu)
    output_musa = dit_musa(x_musa, timestep_musa, y_musa)
    
    print(f"dit_cpu Shape {output_cpu.shape}\n {output_cpu}\n")
    print(f"dit_musa Shape {output_musa.shape}\n {output_musa}\n")
    
    output_cpu.mean().backward()
    output_musa.mean().backward()
    
    assert_close(output_cpu, output_musa, check_device=False)

if __name__ == "__main__": 
    device = "musa"
    # test_ditblock(device) # pass
    # test_ditblock_correctness(device) # remain percision error
    test_dit(device) # pass
    # test_dit_correctness(device)
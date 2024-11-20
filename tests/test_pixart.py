import copy
import torch
import torch.nn as nn
import torch_musa
import sys
from torch.testing import assert_close
from opensora.models.pixart.pixart import PixArtBlock, PixArtMS
from opensora.models.pixart import PixArt


def test_pixartt_block():
    device = torch.device("musa")
    dtype = torch.float32
    torch.manual_seed(1024)
    B, N, C = 4, 64, 128
    x = torch.randn(B, N, C).to(device=device, dtype=dtype)
    y = torch.randn(B, N, C).to(device=device, dtype=dtype)
    mask = [2, 10, 8, 16]
    timestep = torch.randn(B, 6, C).to(device=device, dtype=dtype)  
    
    # Test PixArtBlock: Pass
    pixart_block = PixArtBlock(hidden_size=C, num_heads=N).to(device=device)  
    output = pixart_block(x=x, y=y, t=timestep, mask=mask)
    print(f"PixArtBlock shape {output.shape}\n {output}\n")
    output.mean().backward()
    
    # Test PixArt
    B, N, C, H, W  = 1, 4, 4096, 32, 32 #  1, 4, 4096, 32, 32
    x = torch.randn(B, N, C, H, W ).to(device=device, dtype=dtype) # (N, C, H, W)
    y = torch.randn(N, 1, 120, C).to(device=device, dtype=dtype) # (N, 1, 120, C) tensor of class labels
    timestep = torch.randn(N,).to(device=device, dtype=dtype)  # (N,) tensor of diffusion timesteps
    # mask = [2, 10, 8, 16]
    
    pixart = PixArt(input_size=(C, H, W)).to(device=device) 
    output = pixart(x=x, y=y, timestep=timestep, mask=mask) 
    print(f"PixArt shape {output.shape}\n {output}\n")
    output.mean().backward()
    
    
    # Test PixArtMS
    pixart_ms = PixArtMS(input_size=(C, H, W)).to(device=device) 
    output = pixart_ms(x=x, y=y, timestep=timestep, mask=mask)  
    print(f"PixArt shape {output.shape}\n {output}\n")
    output.mean().backward()


def test_pixart_block_correctness():
    device = torch.device("musa")
    dtype = torch.float32
    torch.manual_seed(1024)
    B, N, C = 4, 64, 128
    x_cpu = torch.randn(B, N, C).to(dtype=dtype)
    y_cpu = torch.randn(B, N, C).to(dtype=dtype)
    mask_cpu = [2, 10, 8, 16]
    timestep_cpu = torch.randn(B, 6, C).to(dtype=dtype)  

    x_musa = copy.deepcopy(x_cpu).to(device=device)
    y_musa = copy.deepcopy(y_cpu).to(device=device)
    timestep_musa = copy.deepcopy(timestep_cpu).to(device=device)
    
    pixart_block_cpu = PixArtBlock(hidden_size=C, num_heads=N)
    pixart_block_musa = copy.deepcopy(pixart_block_cpu).to(device=device)
    
    output_cpu = pixart_block_cpu(x=x_cpu, y=y_cpu, t=timestep_cpu, mask=mask_cpu)
    output_musa = pixart_block_musa(x=x_musa, y=y_musa, t=timestep_musa, mask=mask_cpu)
    
    assert_close(output_cpu, output_musa, check_device=False)


def test_pixart():
    device = torch.device("musa")
    dtype = torch.float16
    torch.manual_seed(1024)
    B, C, T, H, W = 1, 4, 64, 16, 16
    N_token = 120
    # x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
    # t: (N,) tensor of diffusion timesteps
    # y: (N, 1, 120, C) tensor of class labels
    
    x = torch.randn(B, C, T, H, W, dtype=dtype).to(device)  # (B, C, T, H, W)
    x.requires_grad = True
    y = torch.randn(B, 1, N_token, 4096, dtype=dtype).to(device)  #  [B, 1, N_token, C]
    y.requires_grad = True
    timestep = torch.randn(B, dtype=dtype).to(device) 
    # mask = torch.randn(B, N_token, dtype=dtype).to(device)  # [B, N_token]
    mask = None
    
    pixart = PixArt(input_size=(16, 32, 32)).to(device=device) 
    
    x_pixart = pixart(x=x, timestep=timestep, y=y, mask=mask)
    
    print(f"Pixart Shape {x_pixart.shape}\n {x_pixart}\n")
    
    x_pixart.mean().backward()


def test_pixart_correctness():
    device = torch.device("musa")
    dtype = torch.float16
    torch.manual_seed(1024)
    B, C, T, H, W = 1, 4, 64, 16, 16
    N_token = 120
    # x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
    # t: (N,) tensor of diffusion timesteps
    # y: (N, 1, 120, C) tensor of class labels
    
    # CPU
    x_cpu = torch.randn(B, C, T, H, W, dtype=dtype)  # (B, C, T, H, W)
    x_cpu.requires_grad = True
    y_cpu = torch.randn(B, 1, N_token, 4096, dtype=dtype)  #  [B, 1, N_token, C]
    y_cpu.requires_grad = True
    timestep_cpu = torch.randn(B, dtype=dtype)
    # mask = torch.randn(B, N_token, dtype=dtype).to(device)  # [B, N_token]
    mask = None
    
    # MUSA
    x_musa = copy.deepcopy(x_cpu).to(device=device)
    y_musa = copy.deepcopy(y_cpu).to(device=device)
    timestep_musa = copy.deepcopy(timestep_cpu).to(device=device)
    
    
    pixart_cpu = PixArt(input_size=(16, 32, 32)) 
    
    pixart_musa = copy.deepcopy(pixart_cpu).to(device=device)
    
    x_pixart_cpu = pixart_cpu(x=x_cpu, timestep=timestep_cpu, y=y_cpu, mask=mask)
    x_pixart_musa = pixart_musa(x=x_musa, timestep=timestep_musa, y=y_musa, mask=mask)
    
    assert_close(x_pixart_cpu, x_pixart_musa, check_device=False)





if __name__ == "__main__":
    # test_pixartt_block()
    # test_pixart_block_correctness()
    
    test_pixart()
    test_pixart_correctness()
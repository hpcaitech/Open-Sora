import copy
import torch
import torch.nn as nn
import torch_musa
import torch.distributed as dist
from torch.optim import Adam, AdamW
import torch.nn.functional as F
from torch.testing import assert_close
from opensora.models.stdit.stdit2 import STDiT2Block, STDiT2, STDiT2_XL_2
from opensora.acceleration.parallel_states import set_sequence_parallel_group

def setup_param_groups(model: nn.Module) -> list:
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.1,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters



def test_stditblock(device):
    device = torch.device(device)
    torch.manual_seed(1024)
    dtype = torch.float32
    
    B, N, C = 4, 64, 256
    
    stdit_block = STDiT2Block(hidden_size=256, num_heads=8).to(device)
    
    x = torch.randn(B, N, C, dtype=dtype).to(device)  # (B, C, T, H, W)
    x.requires_grad = True
    y = torch.randn(B, N, C, dtype=dtype).to(device)  #  [B, 1, N_token, C]
    y.requires_grad = True
    y.retain_grad()
    timestep = torch.randn(B, 6, dtype=dtype).to(device) 
    temp_timestep = torch.randn(B, 3, dtype=dtype).to(device) 
    
    output = stdit_block(x=x, y=y, t=timestep, t_tmp=temp_timestep,T=8, S=8)
    print(f"stdit_block Shape {output.shape}\n {output}\n")
    
    output.mean().backward()

# stditblock correctness test
def test_stditblock_correctness(device):
    device = torch.device(device)
    torch.manual_seed(1024)
    dtype = torch.float32
    B, N, C = 4, 64, 256
    
    x_cpu = torch.randn(B, N, C).to(dtype=dtype)
    x_cpu.requires_grad = True
    y_cpu = torch.randn(B, N, C).to(dtype=dtype)
    y_cpu.requires_grad = True
    timestep_cpu = torch.randn(B, 6, dtype=dtype)
    temp_timestep_cpu = torch.randn(B, 3, dtype=dtype)
    
    x_musa = copy.deepcopy(x_cpu).to(device=device)
    y_musa = copy.deepcopy(y_cpu).to(device=device)
    timestep_musa = copy.deepcopy(timestep_cpu).to(device=device)
    temp_timestep_cpu_musa = copy.deepcopy(temp_timestep_cpu).to(device=device)
    
    dit_block_cpu = STDiT2Block(hidden_size=256, num_heads=8)
    dit_block_musa = copy.deepcopy(dit_block_cpu).to(device=device)
    
    # check param same
    for (name_cpu, param_cpu), (name_musa, param_musa) in zip(dit_block_cpu.named_parameters(), dit_block_musa.named_parameters()):
        assert_close(param_cpu, param_musa, check_device=False)
        print(f"{name_cpu}, {name_musa} pass")

    
    output_cpu = dit_block_cpu(x=x_cpu, y=y_cpu, t=timestep_cpu, t_tmp=temp_timestep_cpu,T=8, S=8)
    output_musa = dit_block_musa(x=x_musa, y=y_musa, t=timestep_musa, t_tmp=temp_timestep_cpu_musa,T=8, S=8)
    
    print(f"stdit_block_cpu Shape {output_cpu.shape}\n {output_cpu}\n")
    print(f"stdit_block_musa Shape {output_musa.shape}\n {output_musa}\n")
    
    output_cpu.mean().backward()
    output_musa.mean().backward()
    
    assert_close(output_cpu, output_musa, check_device=False)
    
    
def test_stdit(device):
    
    device = torch.device(device)
    torch.manual_seed(1024)
    
    # N, T, D = 4, 64, 256
    B, C, T, H, W = 1, 4, 64, 16, 16
    N_token = 120
    caption_channels = 4096
    device = torch.device(device)
    dtype = torch.float32
    # Variational Auto-Encoder
    stdit = STDiT2(num_heads=4,input_size=(16, 32, 32)).to(device)
    
    x = torch.randn(B, C, T, H, W, dtype=dtype).to(device)  # (B, C, T, H, W)
    x.requires_grad = True
    y = torch.randn(B, 1, N_token, 4096, dtype=dtype).to(device)   #  [B, caption_channels=512]
    timestep = torch.randn(B, dtype=dtype).to(device) #  [B, ]
    height = torch.randn(B, dtype=dtype).to(device) 
    width = torch.randn(W, dtype=dtype).to(device)
    ar = torch.randn(B, dtype=dtype).to(device) 
    num_frames = torch.randn(B, dtype=dtype).to(device) 
    fps = torch.randn(B, dtype=dtype).to(device) 
    mask = torch.randn(B, N_token, dtype=dtype).to(device)  # [B, N_token]
    
    # mask = None
    
    x_stdit = stdit(x=x, timestep=timestep, y=y, mask=mask,height=height, width=width, ar=ar, num_frames=num_frames, fps=fps)

    print(f"STDiT2 Shape {x_stdit.shape}\n {x_stdit}\n")
    
    x_stdit.mean().backward()


# TODO
def test_stdit_correctness(device):
    device = torch.device(device)
    torch.manual_seed(1024)
    
    # N, T, D = 4, 64, 256
    B, C, T, H, W = 1, 4, 64, 16, 16
    N_token = 120
    device = torch.device(device)
    dtype = torch.float32
    # Variational Auto-Encoder
    stdit_cpu = STDiT2(input_size=(16, 32, 32))
    stdit_musa = copy.deepcopy(stdit_cpu).to(device=device)
    
    # x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
    # timestep (torch.Tensor): diffusion time steps; of shape [B]
    # y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
    # mask (torch.Tensor): mask for selecting prompt tokens; of shape [B, N_token]

    x_cpu = torch.randn(B, C, T, H, W, dtype=dtype)  # (B, C, T, H, W)
    x_cpu.requires_grad = True
    y_cpu = torch.randn(B, 1, N_token, 4096, dtype=dtype)  #  [B, 1, N_token, C]
    y_cpu.requires_grad = True
    timestep_cpu = torch.randn(B, dtype=dtype) 
    mask_cpu = torch.randn(B, N_token, dtype=dtype) # [B, N_token]
    # mask_cpu = None
    
    x_musa = copy.deepcopy(x_cpu).to(device=device)
    y_musa = copy.deepcopy(y_cpu).to(device=device)
    timestep_musa = copy.deepcopy(timestep_cpu).to(device=device)
    mask_musa = copy.deepcopy(mask_cpu).to(device=device)
    # mask_musa = None
    
    x_stdit_cpu = stdit_cpu(x=x_cpu, timestep=timestep_cpu, y=y_cpu, mask=mask_cpu)
    x_stdit_musa = stdit_musa(x=x_musa, timestep=timestep_musa, y=y_musa, mask=mask_musa)

    print(f"STDiT2 Shape {x_stdit_cpu.shape}\n {x_stdit_cpu}\n")
    print(f"STDiT2 musa Shape {x_stdit_musa.shape}\n {x_stdit_musa}\n")
    
    x_stdit_cpu.mean().backward()
    x_stdit_musa.mean().backward()


def test_stdit_xl_2(device):
    device = torch.device(device)
    torch.manual_seed(1024)
    
    # N, T, D = 4, 64, 256
    B, C, T, H, W = 1, 4, 4, 16, 16 # T=4 
    N_token = 120
    caption_channels = 4096
    device = torch.device(device)
    dtype = torch.float32
    # stdit_xl_2 = STDiT2_XL_2(from_pretrained="/home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stdit/OpenSora-STDiT-v1-16x256x256/").to(device)
    stdit_xl_2 = STDiT2_XL_2(from_pretrained="/home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stdit/OpenSora/OpenSora-v1-16x256x256.pth").to(device)
    
    
    x = torch.randn(B, C, T, H, W, dtype=dtype).to(device)  # (B, C, T, H, W)
    x.requires_grad = True
    y = torch.randn(B, 1, N_token, 4096, dtype=dtype).to(device)   #  [B, caption_channels=512]
    timestep = torch.randn(B, dtype=dtype).to(device) #  [B, ]
    height = torch.randn(B, dtype=dtype).to(device) 
    width = torch.randn(W, dtype=dtype).to(device)
    ar = torch.randn(B, dtype=dtype).to(device) 
    num_frames = torch.randn(B, dtype=dtype).to(device) 
    fps = torch.randn(B, dtype=dtype).to(device) 
    mask = torch.randn(B, N_token, dtype=dtype).to(device)  # [B, N_token]
    # mask = None
    
    x_stdit = stdit_xl_2(x=x, timestep=timestep, y=y, mask=mask,height=height, width=width, ar=ar, num_frames=num_frames, fps=fps)

    print(f"STDiT Shape {x_stdit.shape}\n {x_stdit}\n")
    
    x_stdit.mean().backward()
    

def test_stdit_xl_2_step(device):
    device = torch.device(device)
    torch.manual_seed(1024)
    
    # N, T, D = 4, 64, 256
    B, C, T, H, W = 1, 4, 4, 16, 16 # T=4 
    N_token = 120
    caption_channels = 4096
    device = torch.device(device)
    dtype = torch.float32
    
    # ==============================
    # Dit Model Init
    # ==============================
    stdit_xl_2 = STDiT2_XL_2(from_pretrained="/home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stdit/OpenSora/OpenSora-v1-16x256x256.pth").to(device)
    model_param = setup_param_groups(stdit_xl_2)
    
    # ==============================
    # Optimizer Init
    # ==============================
    optimizer = Adam(model_param)
    
    # ==============================
    # Dit Data Init
    # ==============================
    # x: video after vae encoder; y: text after t5 encoder
    x = torch.randn(B, C, T, H, W, dtype=dtype).to(device)  # (B, C, T, H, W)
    x.requires_grad = True
    y = torch.randn(B, 1, N_token, 4096, dtype=dtype).to(device)   #  [B, caption_channels=512]
    timestep = torch.randn(B, dtype=dtype).to(device) #  [B, ]
    height = torch.randn(B, dtype=dtype).to(device) 
    width = torch.randn(W, dtype=dtype).to(device)
    ar = torch.randn(B, dtype=dtype).to(device) 
    num_frames = torch.randn(B, dtype=dtype).to(device) 
    fps = torch.randn(B, dtype=dtype).to(device) 
    mask = torch.randn(B, N_token, dtype=dtype).to(device)  # [B, N_token]
    # mask = None
    
    # ==============================
    # Perform Fwd/Bwd
    # ==============================
    x_stdit = stdit_xl_2(x=x, timestep=timestep, y=y, mask=mask,height=height, width=width, ar=ar, num_frames=num_frames, fps=fps) 
    x_stdit.mean().backward() # get grad
    
    
    # ==============================
    # Perform Optim Step
    # ==============================
    optimizer.step()
    optimizer.zero_grad()
    
    # print(f"Param after step:\n {model_param}")
 

if __name__ == "__main__":
    device = "musa"

    # test_stditblock(device)
    # test_stditblock_correctness(device)
    
    # test_stdit(device)
    # test_stdit_correctness(device)
    
    # test_stdit_xl_2(device)
    test_stdit_xl_2_step(device)
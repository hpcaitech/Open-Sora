import copy
import torch
import torch.nn as nn
import torch_musa
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.testing import assert_close

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from opensora.models.stdit.stdit import STDiTBlock, STDiT, STDiT_XL_2
from opensora.models.stdit.stdit2 import STDiT2
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


def test_stdit_xl_2_booster_step(device):
    config = {}
    colossalai.launch(config=config, rank=0, world_size=1, host="localhost", port=31806 , backend="mccl")
    
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
    stdit_xl_2 = STDiT_XL_2(from_pretrained="/home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stdit/OpenSora/OpenSora-v1-16x256x256.pth").to(device)
    model_param = setup_param_groups(stdit_xl_2)
    
    # ==============================
    # Optimizer Init
    # ==============================
    optimizer = Adam(model_param)
    
    # ==============================
    # Plugin & Booster Init
    # ==============================
    plugin = LowLevelZeroPlugin(
            stage=2,
            precision='fp32',
            initial_scale=2**16,
            max_norm=1.0,
        )
    booster = Booster(plugin=plugin)
    criterion = lambda x: x.mean()
    stdit_xl_2, optimizer, criterion, _, _ = booster.boost(stdit_xl_2, optimizer, criterion)

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
    x_stdit = stdit_xl_2(x=x, timestep=timestep, y=y, mask=mask) 
    # x_stdit.mean().backward() # get grad
    optimizer.backward(x_stdit.sum())
    
    # ==============================
    # Perform Optim Step
    # ==============================
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"Param after step:\n {model_param}")
  

if __name__ == "__main__":
    device = "musa"
    test_stdit_xl_2_booster_step(device)
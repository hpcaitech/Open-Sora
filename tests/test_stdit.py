import copy
import torch
import torch.nn as nn
import torch_musa
import thop
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.testing import assert_close
from colossalai.testing import spawn
import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from opensora.models.stdit.stdit import STDiTBlock, STDiT, STDiT_XL_2
from opensora.models.stdit.stdit2 import STDiT2
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.utils.train_utils import set_seed
# from .test_assert_closed import assert_tensor, assert_param

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
    
    stdit_block = STDiTBlock(hidden_size=256, num_heads=8, d_s=8, d_t=8).to(device)
    
    x = torch.randn(B, N, C, dtype=dtype).to(device)  # (B, C, T, H, W)
    x.requires_grad = True
    y = torch.randn(B, N, C, dtype=dtype).to(device)  #  [B, 1, N_token, C]
    y.requires_grad = True
    y.retain_grad()
    timestep = torch.randn(B, 6, dtype=dtype).to(device) 
    
    output = stdit_block(x, y, timestep)
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
    
    x_musa = copy.deepcopy(x_cpu).to(device=device)
    y_musa = copy.deepcopy(y_cpu).to(device=device)
    timestep_musa = copy.deepcopy(timestep_cpu).to(device=device)
    
    dit_block_cpu = STDiTBlock(hidden_size=256, num_heads=8, d_s=8, d_t=8)
    dit_block_musa = copy.deepcopy(dit_block_cpu).to(device=device)
    
    # check param same
    for (name_cpu, param_cpu), (name_musa, param_musa) in zip(dit_block_cpu.named_parameters(), dit_block_musa.named_parameters()):
        assert_close(param_cpu, param_musa, check_device=False)
        print(f"{name_cpu}, {name_musa} pass")

    
    output_cpu = dit_block_cpu(x_cpu, y_cpu, timestep_cpu)
    output_musa = dit_block_musa(x_musa, y_musa, timestep_musa)
    
    print(f"stdit_block_cpu Shape {output_cpu.shape}\n {output_cpu}\n")
    print(f"stdit_block_musa Shape {output_musa.shape}\n {output_musa}\n")
    
    output_cpu.mean().backward()
    output_musa.mean().backward()
    
    assert_close(output_cpu, output_musa, check_device=False)
    
    
def test_stdit(device):
    # # q = torch.rand(1, 4096, 4, 32, dtype=torch.float16, device="musa")
    # # k = torch.rand(1, 120,  4, 32, dtype=torch.float16, device="musa")
    # # v = torch.rand(1, 120,  4, 32, dtype=torch.float16, device="musa")
    
    # q = torch.rand(1, 4, 4096, 32, dtype=torch.float16, device="musa")
    # k = torch.rand(1, 4,  120, 32, dtype=torch.float16, device="musa")
    # v = torch.rand(1, 4,  120, 32, dtype=torch.float16, device="musa")
    
    # # mask = torch.rand(128, 64, dtype=torch.float16, device="musa")
    # # print(mask)
    # out = F.scaled_dot_product_attention(q,k,v, attn_mask=None)
    # print(out)
    
    
    device = torch.device(device)
    torch.manual_seed(1024)
    
    # N, T, D = 4, 64, 256
    B, C, T, H, W = 1, 4, 64, 16, 16
    N_token = 120
    caption_channels = 4096
    device = torch.device(device)
    dtype = torch.float32
    # Variational Auto-Encoder
    stdit = STDiT(hidden_size=128, num_heads=4,input_size=(16, 32, 32)).to(device)
    
    # x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
    # timestep (torch.Tensor): diffusion time steps; of shape [B]
    # y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
    # mask (torch.Tensor): mask for selecting prompt tokens; of shape [B, N_token]

    # x = torch.randn(B, C, T, H, W, dtype=dtype).to(device)  # (B, C, T, H, W)
    # x.requires_grad = True
    # y = torch.randn(B, 1, N_token, C, dtype=dtype).to(device)  #  [B, 1, N_token, C]
    # y.requires_grad = True
    # timestep = torch.randn(B, dtype=dtype).to(device) 
    
    x = torch.randn(B, C, T, H, W, dtype=dtype).to(device)  # (B, C, T, H, W)
    x.requires_grad = True
    y = torch.randn(B, 1, N_token, 4096, dtype=dtype).to(device)   #  [B, caption_channels=512]
    timestep = torch.randn(B, dtype=dtype).to(device) #  [B, ]
    
    mask = torch.randn(B, N_token, dtype=dtype).to(device)  # [B, N_token]
    # mask = None
    
    x_stdit = stdit(x=x, timestep=timestep, y=y, mask=mask)

    print(f"STDiT Shape {x_stdit.shape}\n {x_stdit}\n")
    
    x_stdit.mean().backward()


def test_stdit_correctness(device):
    device = torch.device(device)
    torch.manual_seed(1024)
    
    # N, T, D = 4, 64, 256
    B, C, T, H, W = 1, 4, 64, 16, 16
    N_token = 120
    device = torch.device(device)
    dtype = torch.float32
    # Variational Auto-Encoder
    stdit_cpu = STDiT(input_size=(16, 32, 32))
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

    print(f"STDiT Shape {x_stdit_cpu.shape}\n {x_stdit_cpu}\n")
    print(f"STDiT musa Shape {x_stdit_musa.shape}\n {x_stdit_musa}\n")
    
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
    
    # stdit_xl_2 = STDiT_XL_2(from_pretrained="/home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stdit/OpenSora/OpenSora-v1-16x256x256.pth").to(device)
    stdit_xl_2 = STDiT_XL_2(from_pretrained="./pretrained_models/PixArt-alpha/PixArt-XL-2-512x512.pth").to(device)
    
    
    x = torch.randn(B, C, T, H, W, dtype=dtype).to(device)  # (B, C, T, H, W)
    x.requires_grad = True
    y = torch.randn(B, 1, N_token, 4096, dtype=dtype).to(device)   #  [B, caption_channels=512]
    timestep = torch.randn(B, dtype=dtype).to(device) #  [B, ]
    mask = torch.randn(B, N_token, dtype=dtype).to(device)  # [B, N_token] # Method 1
    # mask = torch.randn(256, N_token, dtype=dtype).to(device)  # [B, N_token] # Method 2
    # mask = None
    x_stdit = stdit_xl_2(x=x, timestep=timestep, y=y, mask=mask)

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
    stdit_xl_2 = STDiT_XL_2(from_pretrained="/home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stdit/OpenSora/OpenSora-v1-16x256x256.pth").to(device)
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
    x_stdit = stdit_xl_2(x=x, timestep=timestep, y=y, mask=mask) 
    x_stdit.mean().backward() # get grad
    
    
    # ==============================
    # Perform Optim Step
    # ==============================
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"Param after step:\n {model_param}")
    
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
    # dtype = torch.float32
    # dtype = torch.float16
    dtype = torch.bfloat16
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
            # precision='fp32',
            # precision='fp16',
            precision='bf16',
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
    
    stdit_flops, stdit_params = thop.profile(model=stdit_xl_2, inputs=(x, timestep ,y, mask))
    print(f"stdit_flops {stdit_flops}; stdit_params {stdit_params}")
    # ==============================
    # Perform Fwd/Bwd
    # ==============================
    x_stdit = stdit_xl_2(x=x, timestep=timestep, y=y, mask=mask) 
    # print(f"x_stdit output {x_stdit}")
    # print(f"loss {x_stdit.mean()}")
    # x_stdit.mean().backward() # get grad
    loss = optimizer.backward(x_stdit.sum())
    # print(f"loss {loss}")
    # ==============================
    # Perform Optim Step
    # ==============================
    optimizer.step()
    optimizer.zero_grad()
    
    # print(f"Param after step:\n {model_param}")
    
def run_seq_parallel_stdit(rank, world_size):
    device="musa"
    dtype = torch.bfloat16
    torch.manual_seed(1024)
    set_sequence_parallel_group(dist.group.WORLD)
    
    model = STDiT_XL_2(
        space_scale=0.5,
        time_scale=1.0,
        enable_sequence_parallelism = True,
        enable_flashattn=True,
        enable_layernorm_kernel=False,
    )
    model.to(device, dtype)
    # B, C, T, H, W = 16, 4, 16, 32, 32 # T=4 
    B, C, T, H, W = 1, 4, 4, 16, 16
    N_token = 120
    caption_channels = 4096
    x = torch.randn(B, C, T, H, W, dtype=dtype).to(device)  # (B, C, T, H, W)
    x.requires_grad = True
    y = torch.randn(B, 1, N_token, 4096, dtype=dtype).to(device)   #  [B, caption_channels=512]
    timestep = torch.randint(0, 1000, (x.shape[0],), device=device)
    mask = None  # [B, N_token] # Method 1
   
    x_stdit = model(x=x, timestep=timestep, y=y, mask=mask)

    print(f"STDiT seq parallel Shape {x_stdit.shape}\n {x_stdit}\n")
    
    x_stdit.mean().backward()


def run_seq_parallel_stditblk(rank, world_size):
    device = "musa"
    torch.manual_seed(1024)
    dtype = torch.bfloat16
    torch.manual_seed(1024)
    set_sequence_parallel_group(dist.group.WORLD)
    
    B, N, C = 4, 32, 256
    
    stdit_block = STDiTBlock(
        hidden_size=256, 
        num_heads=8, d_s=8, d_t=8,
        enable_flashattn=True,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=True,
        ).to(device)
    
    x = torch.randn(B, N, C, dtype=dtype).to(device)  # (B, C, T, H, W)
    x.requires_grad = True
    y = torch.randn(B, N, C, dtype=dtype).to(device)  #  [B, 1, N_token, C]
    y.requires_grad = True
    y.retain_grad()
    timestep = torch.randn(B, 6, dtype=dtype).to(device) 
    
    output = stdit_block(x, y, timestep)
    print(f"stdit_block Shape {output.shape}\n {output}\n")
    
    output.mean().backward()

    
    

def run_dist(rank, world_size, port):
    colossalai.launch({}, rank=rank, world_size=world_size, host="localhost", port=port, backend="mccl")
    run_seq_parallel_stdit(rank, world_size)
    # run_seq_parallel_stditblk(rank, world_size)
    
def test_seq_parallel_stdit():
    spawn(run_dist, nprocs=2)


def test_stdit_single_op(device):
    device = torch.device(device)
    # torch.manual_seed(1024)
    set_seed(1024)
    # N, T, D = 4, 64, 256
    B, C, T, H, W = 1, 4, 4, 16, 16 # T=4 
    N_token = 120
    caption_channels = 4096
    device = torch.device(device)
    dtype = torch.float32
    
    # stdit_xl_2 = STDiT_XL_2(from_pretrained="/home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stdit/OpenSora/OpenSora-v1-16x256x256.pth").to(device)
    # stdit_xl_2 = STDiT_XL_2(from_pretrained="./pretrained_models/PixArt-alpha/PixArt-XL-2-512x512.pth").to(device)
    stdit_xl_2 = STDiT_XL_2(from_pretrained="./pretrained_models/stdit/OpenSora/OpenSora-v1-16x256x256.pth").to(device)
    
    x = torch.randn(B, C, T, H, W, dtype=dtype).to(device)  # (B, C, T, H, W)
    x.requires_grad = True
    y = torch.randn(B, 1, N_token, 4096, dtype=dtype).to(device)   #  [B, caption_channels=512]
    timestep = torch.randn(B, dtype=dtype).to(device) #  [B, ]
    mask = torch.randn(B, N_token, dtype=dtype).to(device)  # [B, N_token] # Method 1
    # mask = torch.randn(256, N_token, dtype=dtype).to(device)  # [B, N_token] # Method 2
    # mask = None
    torch.save(x , f"./dataset/assert_closed/musa_tensor/single_op_stdit_input.txt")
    torch.save(stdit_xl_2.state_dict() , f"./dataset/assert_closed/musa_tensor/single_op_stdit_param_init.txt")
    
    x_stdit = stdit_xl_2(x=x, timestep=timestep, y=y, mask=mask)
    
    torch.save(x_stdit , f"./dataset/assert_closed/musa_tensor/single_op_stdit_output.txt")
    
    # x_stdit.mean().backward()
    # torch.save(stdit_xl_2.state_dict() , f"./dataset/assert_closed/musa_tensor/single_op_stdit_param_bwd.txt")


def assert_stdit_single_op():
    # assert input
    model_input_musa = torch.load(f"./dataset/assert_closed/musa_tensor/single_op_stdit_input.txt", map_location=torch.device('musa'))
    model_input_torch = torch.load(f"./dataset/assert_closed/torch_tensor/single_op_stdit_input.txt", map_location=torch.device('musa'))
    assert_close(model_input_musa, model_input_torch)
    print(f"stdit input assert close pass;")
    
    # assert model param init
    model_param_init_musa = torch.load(f"./dataset/assert_closed/musa_tensor/single_op_stdit_param_init.txt", map_location=torch.device('musa'))
    model_param_init_torch = torch.load(f"./dataset/assert_closed/torch_tensor/single_op_stdit_param_init.txt", map_location=torch.device('musa'))
    assert_close(model_param_init_musa, model_param_init_torch)
    print(f"stdit param init assert close pass;")
    
    # assert output
    model_output_musa = torch.load(f"./dataset/assert_closed/musa_tensor/single_op_stdit_output.txt", map_location=torch.device('musa'))
    model_output_torch = torch.load(f"./dataset/assert_closed/torch_tensor/single_op_stdit_output.txt", map_location=torch.device('musa'))
    assert_close(model_output_musa, model_output_torch)
    print(f"stdit output assert close pass;")
    
    # assert model param bwd
    model_param_bwd_musa = torch.load(f"./dataset/assert_closed/musa_tensor/single_op_stdit_param_bwd.txt", map_location=torch.device('musa'))
    model_param_bwd_torch = torch.load(f"./dataset/assert_closed/torch_tensor/single_op_stdit_param_bwd.txt", map_location=torch.device('musa'))
    assert_close(model_param_bwd_musa, model_param_bwd_torch)
    print(f"stdit param bwd assert close pass;")
    

if __name__ == "__main__":
    device = "musa"

    # test_stditblock(device)
    # test_stditblock_correctness(device)
    
    # test_stdit(device)
    # test_stdit_correctness(device)
    
    # test_stdit_xl_2(device)
    # test_stdit_xl_2_step(device)
    # test_stdit_xl_2_booster_step(device)
    # test_stdit_single_op(device)
    # assert_stdit_single_op()
    test_seq_parallel_stdit()
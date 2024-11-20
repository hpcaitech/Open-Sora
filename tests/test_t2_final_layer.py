import copy
import torch
import torch_musa
from torch.testing import assert_close
from opensora.models.layers.blocks import T2IFinalLayer

def test_t2final_layer(device):
    dtype = torch.bfloat16
    B, T, S, C = 2, 64, 64, 128
    x = torch.randn(B, T * S, C , device=device, dtype=dtype).requires_grad_()
    # x_mask = torch.randn(B, T, device=device, dtype=dtype).requires_grad_()
    t = torch.randn(B, C, device=device, dtype=dtype) / 128**0.5
    t0 = torch.randn(B, C, device=device, dtype=dtype) / 128**0.5
    # x: [B, (T, S), C]
    # mased_x: [B, (T, S), C]
    # x_mask: [B, T]
    t2_final_layer = T2IFinalLayer(
        hidden_size=C,
        num_patch=4,
        out_channels=256,
    ).to(device=device, dtype=dtype)
    output = t2_final_layer(
        x=x, 
        t=t,
        # x_mask=x_mask,
        t0=t0,
        T=T,
        S=S)
    print(f"Shape {output.shape}\n {output}\n")

# TODO: distributed test; may not;
# TODO: correctness test 
def test_t2final_layer_correctness():
    dtype = torch.bfloat16
    device="musa"
    torch.manual_seed(1024)
    
    B, T, S, C = 2, 64, 64, 128
    x_cpu = torch.randn(B, T * S, C , dtype=dtype).requires_grad_()
    # x_mask = torch.randn(B, T, device=device, dtype=dtype).requires_grad_()
    t_cpu = torch.randn(B, C, dtype=dtype) / 128**0.5
    t0_cpu = torch.randn(B, C, dtype=dtype) / 128**0.5
    # x: [B, (T, S), C]
    # mased_x: [B, (T, S), C]
    # x_mask: [B, T]
    x_musa = copy.deepcopy(x_cpu).to(device=device)
    # x_mask = torch.randn(B, T, device=device, dtype=dtype).requires_grad_()
    t_musa = copy.deepcopy(t_cpu).to(device=device)
    t0_musa = copy.deepcopy(t0_cpu).to(device=device)

    
    t2_final_layer_cpu = T2IFinalLayer(
        hidden_size=C,
        num_patch=4,
        out_channels=256,
    ).to(dtype=dtype)
    
    t2_final_layer_musa = copy.deepcopy(t2_final_layer_cpu).to(device=device)
    
    output_cpu = t2_final_layer_cpu(
        x=x_cpu, 
        t=t_cpu,
        # x_mask=x_mask,
        t0=t0_cpu,
        T=T,
        S=S)
    
    output_musa = t2_final_layer_musa(
        x=x_musa, 
        t=t_musa,
        # x_mask=x_mask,
        t0=t0_musa,
        T=T,
        S=S)
    
    assert_close(output_cpu, output_musa, check_device=False)


if __name__ == "__main__":
    print("Test T2IFinalLayer")
    test_t2final_layer("musa")
    
    print("Test T2IFinalLayer Correctness")
    test_t2final_layer_correctness()
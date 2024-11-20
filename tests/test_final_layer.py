import copy
import torch
import torch_musa
from torch.testing import assert_close
from opensora.models.layers.blocks import FinalLayer

def test_final_layer(device):
    dtype = torch.bfloat16
    x = torch.randn(4, 8, 128, device=device, dtype=dtype).requires_grad_()
    c = torch.randn(4, 128, device=device, dtype=dtype).requires_grad_()
    # Suppose x is (B, N, D), shift is (B, D), scale is (B, D)
    final_layer = FinalLayer(
        hidden_size=128,
        num_patch=4,
        out_channels=256,
    ).to(device=device, dtype=dtype)
    output = final_layer(x, c)
    output.sum().backward()
    print(f"Shape {output.shape}\n {output}\n")


# TODO: distributed test; may not;
# TODO: correctness test 
def test_final_layer_correctness():
    dtype = torch.bfloat16
    device="musa"
    torch.manual_seed(1024)
    
    x_cpu = torch.randn(4, 8, 128, dtype=dtype).requires_grad_()
    c_cpu = torch.randn(4, 128, dtype=dtype).requires_grad_()
    
    x_musa = copy.deepcopy(x_cpu).to(device=device)
    c_musa = copy.deepcopy(c_cpu).to(device=device)
    
    final_layer_cpu = FinalLayer(
        hidden_size=128,
        num_patch=4,
        out_channels=256,
    ).to(dtype=dtype)

    final_layer_musa = copy.deepcopy(final_layer_cpu).to(device=device)
    
    output_cpu = final_layer_cpu(x_cpu, c_cpu)
    output_musa = final_layer_musa(x_musa, c_musa)
    
    assert_close(output_cpu, output_musa, check_device=False)

if __name__ == "__main__":
    print("Test Final Layer")
    test_final_layer("musa")
    
    print("Test Final Layer Correctness")
    test_final_layer_correctness()
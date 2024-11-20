import torch
import torch_musa
import copy
from torch.testing import assert_close


if __name__ == "__main__":
    torch.manual_seed(1024)
    dtype = torch.float64
    shape = 1000

    a = torch.randn(shape, dtype=dtype, requires_grad=True)
    a_musa = copy.deepcopy(a).to(device="musa")
    
    # fwd
    b = torch.cumprod(a, dim=0)
    b_musa = torch.cumprod(a, dim=0)
    # print(f"fwd {b} {b_musa}")
    
    assert_close(b, b_musa, check_device=False)

    # bwd
    b.mean().backward()
    b_musa.mean().backward()
    
    
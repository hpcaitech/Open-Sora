import torch
import torch_musa

a = torch.rand(10, dtype=torch.bfloat16, device='musa')
a.requires_grad=True
b = torch.erf(a)
b.sum().backward()
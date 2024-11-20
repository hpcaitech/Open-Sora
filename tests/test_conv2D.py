import torch
import torch_musa
import torch.nn.functional as F

dtype = dtype=torch.bfloat16 # float16, bfloat16, float32
filters = torch.randn(8, 4, 3, 3).to(dtype=dtype, device='musa')
inputs = torch.randn(1, 4, 5, 5).to(dtype=dtype, device='musa')
inputs.requires_grad=True
res = F.conv2d(inputs, filters, padding=1)

res.mean().backward()
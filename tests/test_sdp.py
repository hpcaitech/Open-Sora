import torch
import torch_musa
import torch.nn.functional as F

query = torch.rand(32, 8, 4096, 64, dtype=torch.bfloat16, device="musa")
key = torch.rand(32, 8, 120, 64, dtype=torch.bfloat16, device="musa")
value = torch.rand(32, 8, 120, 64, dtype=torch.bfloat16, device="musa")
res = F.scaled_dot_product_attention(query,key,value)

print(res)
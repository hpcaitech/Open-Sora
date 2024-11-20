import torch
import torch_musa

# y torch.bfloat16 torch.Size([2, 1, 200, 1152]); x torch.bfloat16 torch.Size([2, 50880, 1152]); mask torch.int64 torch.Size([2, 200])
y = torch.rand(2,1,200,1152, dtype=torch.bfloat16, device='musa')
x = torch.rand(2, 50880, 1152, dtype=torch.bfloat16, device='musa')
mask = torch.randint(1, (2, 200), dtype=torch.int64, device='musa')

y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
import torch
import torch_musa

y_embedding = torch.rand(120, 4096, dtype=torch.bfloat16, device='musa')
caption = torch.randn(2, 1, 120, 4096, dtype=torch.bfloat16, device='musa')
caption = torch.where(y_embedding > 0, y_embedding, caption)
print(f"caption {caption}")
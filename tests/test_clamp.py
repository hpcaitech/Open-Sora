import torch
import torch_musa

# a = torch.randn(1, 4, dtype=torch.float16).musa()
# a = torch.clamp(a, min=-0.5, max=0.5)
# print(a)


# a = torch.randn(4, dtype=torch.bfloat16).musa()
# b = torch.randn(4, dtype=torch.bfloat16).musa()
# a = torch.clamp(a, min=-b, max=b)
# print(a)
dtype = torch.float16 # torch.float16
hidden_states = torch.randn(1, 120, 4096, dtype=dtype).musa()
hidden_states.require_grad=True
clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            ).to(dtype=hidden_states.dtype, device='musa')
print(f"hidden_states {hidden_states.dtype} clamp_value {clamp_value.dtype}")
hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
# print(hidden_states)
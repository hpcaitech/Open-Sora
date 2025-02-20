# modified from
# https://github.com/bornfly-detachment/asymmetric_magvitv2/blob/main/models/utils/util.py


import torch
from einops import rearrange


def to_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        dtype_mapping = {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype not in dtype_mapping:
            raise ValueError
        dtype = dtype_mapping[dtype]
        return dtype
    else:
        raise ValueError


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def video_to_image(func):
    def wrapper(self, x, *args, **kwargs):
        if x.dim() == 5:
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            micro_batch_size_2d = self.micro_batch_size_2d if hasattr(self, "micro_batch_size_2d") else None
            if micro_batch_size_2d is None:
                x = func(self, x, *args, **kwargs)
            else:
                bs = micro_batch_size_2d
                x_out = []
                for i in range(0, x.shape[0], bs):
                    x_bs = x[i : i + bs]
                    x_bs = func(self, x_bs, *args, **kwargs)
                    x_out.append(x_bs)
                x = torch.cat(x_out, dim=0)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x

    return wrapper

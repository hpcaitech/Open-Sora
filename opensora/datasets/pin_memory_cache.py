import threading
from typing import Dict, List, Optional

import torch


class PinMemoryCache:
    force_dtype: Optional[torch.dtype] = None
    min_cache_numel: int = 0
    pre_alloc_numels: List[int] = []

    def __init__(self):
        self.cache: Dict[int, torch.Tensor] = {}
        self.output_to_cache: Dict[int, int] = {}
        self.cache_to_output: Dict[int, int] = {}
        self.lock = threading.Lock()
        self.total_cnt = 0
        self.hit_cnt = 0

        if len(self.pre_alloc_numels) > 0 and self.force_dtype is not None:
            for n in self.pre_alloc_numels:
                cache_tensor = torch.empty(n, dtype=self.force_dtype, device="cpu", pin_memory=True)
                with self.lock:
                    self.cache[id(cache_tensor)] = cache_tensor

    def get(self, tensor: torch.Tensor) -> torch.Tensor:
        """Receive a cpu tensor and return the corresponding pinned tensor. Note that this only manage memory allocation, doesn't copy content.

        Args:
            tensor (torch.Tensor): The tensor to be pinned.

        Returns:
            torch.Tensor: The pinned tensor.
        """
        self.total_cnt += 1
        with self.lock:
            # find free cache
            for cache_id, cache_tensor in self.cache.items():
                if cache_id not in self.cache_to_output and cache_tensor.numel() >= tensor.numel():
                    target_cache_tensor = cache_tensor[: tensor.numel()].view(tensor.shape)
                    out_id = id(target_cache_tensor)
                    self.output_to_cache[out_id] = cache_id
                    self.cache_to_output[cache_id] = out_id
                    self.hit_cnt += 1
                    return target_cache_tensor
        # no free cache, create a new one
        dtype = self.force_dtype if self.force_dtype is not None else tensor.dtype
        cache_numel = max(tensor.numel(), self.min_cache_numel)
        cache_tensor = torch.empty(cache_numel, dtype=dtype, device="cpu", pin_memory=True)
        target_cache_tensor = cache_tensor[: tensor.numel()].view(tensor.shape)
        out_id = id(target_cache_tensor)
        with self.lock:
            self.cache[id(cache_tensor)] = cache_tensor
            self.output_to_cache[out_id] = id(cache_tensor)
            self.cache_to_output[id(cache_tensor)] = out_id
        return target_cache_tensor

    def remove(self, output_tensor: torch.Tensor) -> None:
        """Release corresponding cache tensor.

        Args:
            output_tensor (torch.Tensor): The tensor to be released.
        """
        out_id = id(output_tensor)
        with self.lock:
            if out_id not in self.output_to_cache:
                raise ValueError("Tensor not found in cache.")
            cache_id = self.output_to_cache.pop(out_id)
            del self.cache_to_output[cache_id]

    def __str__(self):
        with self.lock:
            num_cached = len(self.cache)
            num_used = len(self.output_to_cache)
            total_cache_size = sum([v.numel() * v.element_size() for v in self.cache.values()])
        return f"PinMemoryCache(num_cached={num_cached}, num_used={num_used}, total_cache_size={total_cache_size / 1024**3:.2f} GB, hit rate={self.hit_cnt / self.total_cnt:.2f})"

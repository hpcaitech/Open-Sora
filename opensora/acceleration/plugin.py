import random
from typing import Optional

import numpy as np
import torch
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import ProcessGroupMesh
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

DP_AXIS, SP_AXIS = 0, 1


class ZeroSeqParallelPlugin(LowLevelZeroPlugin):
    def __init__(
        self,
        sp_size: int = 1,
        stage: int = 2,
        precision: str = "fp16",
        initial_scale: float = 2**32,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
        max_norm: float = 0.0,
        norm_type: float = 2.0,
        reduce_bucket_size_in_m: int = 12,
        communication_dtype: Optional[torch.dtype] = None,
        overlap_communication: bool = True,
        cpu_offload: bool = False,
        master_weights: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            stage=stage,
            precision=precision,
            initial_scale=initial_scale,
            min_scale=min_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            max_scale=max_scale,
            max_norm=max_norm,
            norm_type=norm_type,
            reduce_bucket_size_in_m=reduce_bucket_size_in_m,
            communication_dtype=communication_dtype,
            overlap_communication=overlap_communication,
            cpu_offload=cpu_offload,
            master_weights=master_weights,
            verbose=verbose,
        )
        self.sp_size = sp_size
        assert self.world_size % sp_size == 0, "world_size must be divisible by sp_size"
        self.dp_size = self.world_size // sp_size
        self.pg_mesh = ProcessGroupMesh(self.dp_size, self.sp_size)
        self.dp_group = self.pg_mesh.get_group_along_axis(DP_AXIS)
        self.sp_group = self.pg_mesh.get_group_along_axis(SP_AXIS)
        self.dp_rank = self.pg_mesh.coordinate(DP_AXIS)
        self.sp_rank = self.pg_mesh.coordinate(SP_AXIS)

    def __del__(self):
        """Destroy the prcess groups in ProcessGroupMesh"""
        self.pg_mesh.destroy_mesh_process_groups()

    def prepare_dataloader(
        self,
        dataset,
        batch_size,
        shuffle=False,
        seed=1024,
        drop_last=False,
        pin_memory=False,
        num_workers=0,
        distributed_sampler_cls=None,
        **kwargs,
    ):
        _kwargs = kwargs.copy()
        distributed_sampler_cls = distributed_sampler_cls or DistributedSampler
        sampler = distributed_sampler_cls(dataset, num_replicas=self.dp_size, rank=self.dp_rank, shuffle=shuffle)

        # Deterministic dataloader
        def seed_worker(worker_id):
            worker_seed = seed
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            random.seed(worker_seed)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            worker_init_fn=seed_worker,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            **_kwargs,
        )

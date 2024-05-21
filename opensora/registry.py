from copy import deepcopy

import torch.nn as nn
from mmengine.registry import Registry


def build_module(module, builder, **kwargs):
    """Build module from config or return the module itself.

    Args:
        module (Union[dict, nn.Module]): The module to build.
        builder (Registry): The registry to build module.
        *args, **kwargs: Arguments passed to build function.

    Returns:
        Any: The built module.
    """
    if module is None:
        return None
    if isinstance(module, dict):
        cfg = deepcopy(module)
        for k, v in kwargs.items():
            cfg[k] = v
        return builder.build(cfg)
    elif isinstance(module, nn.Module):
        return module
    elif module is None:
        return None
    else:
        raise TypeError(f"Only support dict and nn.Module, but got {type(module)}.")


MODELS = Registry(
    "model",
    locations=["opensora.models"],
)

SCHEDULERS = Registry(
    "scheduler",
    locations=["opensora.schedulers"],
)

DATASETS = Registry(
    "dataset",
    locations=["opensora.datasets"],
)

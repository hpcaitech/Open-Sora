import colossalai
import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator

from opensora.acceleration.parallel_states import (
    get_sequence_parallel_group,
    get_tensor_parallel_group,
    set_sequence_parallel_group,
)
from opensora.models.hunyuan_vae.policy import HunyuanVaePolicy
from opensora.models.mmdit.distributed import MMDiTPolicy
from opensora.utils.logger import is_distributed
from opensora.utils.train import create_colossalai_plugin

from .logger import log_message


def set_group_size(plugin_config: dict):
    """
    Set the group size for tensor parallelism and sequence parallelism.

    Args:
        plugin_config (dict): Plugin configuration.
    """
    tp_size = int(plugin_config.get("tp_size", 1))
    sp_size = int(plugin_config.get("sp_size", 1))
    if tp_size > 1:
        assert sp_size == 1
        plugin_config["tp_size"] = tp_size = min(tp_size, torch.cuda.device_count())
        log_message(f"Using TP with size {tp_size}")
    if sp_size > 1:
        assert tp_size == 1
        plugin_config["sp_size"] = sp_size = min(sp_size, torch.cuda.device_count())
        log_message(f"Using SP with size {sp_size}")


def init_inference_environment():
    """
    Initialize the inference environment.
    """
    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)


def get_booster(cfg: dict, ae: bool = False):
    suffix = "_ae" if ae else ""
    policy = HunyuanVaePolicy if ae else MMDiTPolicy

    plugin_type = cfg.get(f"plugin{suffix}", "zero2")
    plugin_config = cfg.get(f"plugin_config{suffix}", {})
    plugin_kwargs = {}
    booster = None
    if plugin_type == "hybrid":
        set_group_size(plugin_config)
        plugin_kwargs = dict(custom_policy=policy)

        plugin = create_colossalai_plugin(
            plugin=plugin_type,
            dtype=cfg.get("dtype", "bf16"),
            grad_clip=cfg.get("grad_clip", 0),
            **plugin_config,
            **plugin_kwargs,
        )
        booster = Booster(plugin=plugin)
    return booster


def get_is_saving_process(cfg: dict):
    """
    Check if the current process is the one that saves the model.

    Args:
        plugin_config (dict): Plugin configuration.

    Returns:
        bool: True if the current process is the one that saves the model.
    """
    plugin_type = cfg.get("plugin", "zero2")
    plugin_config = cfg.get("plugin_config", {})
    is_saving_process = (
        plugin_type != "hybrid"
        or (plugin_config["tp_size"] > 1 and dist.get_rank(get_tensor_parallel_group()) == 0)
        or (plugin_config["sp_size"] > 1 and dist.get_rank(get_sequence_parallel_group()) == 0)
    )
    return is_saving_process

import collections
import importlib
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Sequence
from itertools import repeat
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from colossalai.cluster.dist_coordinator import DistCoordinator

# ======================================================
# Logging
# ======================================================


def is_distributed():
    return os.environ.get("WORLD_SIZE", None) is not None


def is_main_process():
    return not is_distributed() or dist.get_rank() == 0


def get_world_size():
    if is_distributed():
        return dist.get_world_size()
    else:
        return 1


def create_logger(logging_dir=None):
    """
    Create a logger that writes to a log file and stdout.
    """
    if is_main_process():  # real logger
        additional_args = dict()
        if logging_dir is not None:
            additional_args["handlers"] = [
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ]
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            **additional_args,
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def get_logger():
    return logging.getLogger(__name__)


def print_rank(var_name, var_value, rank=0):
    if dist.get_rank() == rank:
        print(f"[Rank {rank}] {var_name}: {var_value}")


def print_0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def create_tensorboard_writer(exp_dir):
    from torch.utils.tensorboard import SummaryWriter

    tensorboard_dir = f"{exp_dir}/tensorboard"
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    return writer


# ======================================================
# String
# ======================================================


def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


def get_timestamp():
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))
    return timestamp


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


class BColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# ======================================================
# PyTorch
# ======================================================


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


def get_model_numel(model: torch.nn.Module) -> Tuple[int, int]:
    num_params = 0
    num_params_trainable = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_trainable += p.numel()
    return num_params, num_params_trainable


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")


def to_ndarray(data):
    if isinstance(data, torch.Tensor):
        return data.numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, Sequence):
        return np.array(data)
    elif isinstance(data, int):
        return np.ndarray([data], dtype=int)
    elif isinstance(data, float):
        return np.array([data], dtype=float)
    else:
        raise TypeError(f"type {type(data)} cannot be converted to ndarray.")


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


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def convert_SyncBN_to_BN2d(model_cfg):
    for k in model_cfg:
        v = model_cfg[k]
        if k == "norm_cfg" and v["type"] == "SyncBN":
            v["type"] = "BN2d"
        elif isinstance(v, dict):
            convert_SyncBN_to_BN2d(v)


def get_topk(x, dim=4, k=5):
    x = to_tensor(x)
    inds = x[..., dim].topk(k)[1]
    return x[inds]


def param_sigmoid(x, alpha):
    ret = 1 / (1 + (-alpha * x).exp())
    return ret


def inverse_param_sigmoid(x, alpha, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2) / alpha


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# ======================================================
# Python
# ======================================================


def count_columns(df, columns):
    cnt_dict = OrderedDict()
    num_samples = len(df)

    for col in columns:
        d_i = df[col].value_counts().to_dict()
        for k in d_i:
            d_i[k] = (d_i[k], d_i[k] / num_samples)
        cnt_dict[col] = d_i

    return cnt_dict


def try_import(name):
    """Try to import a module.

    Args:
        name (str): Specifies what module to import in absolute or relative
            terms (e.g. either pkg.mod or ..mod).
    Returns:
        ModuleType or None: If importing successfully, returns the imported
        module, otherwise returns None.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def transpose(x):
    """
    transpose a list of list
    Args:
        x (list[list]):
    """
    ret = list(map(list, zip(*x)))
    return ret


def all_exists(paths):
    return all(os.path.exists(path) for path in paths)


# ======================================================
# Profile
# ======================================================


class Timer:
    def __init__(self, name, log=False, coordinator: Optional[DistCoordinator] = None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.log = log
        self.coordinator = coordinator

    @property
    def elapsed_time(self):
        return self.end_time - self.start_time

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.coordinator is not None:
            self.coordinator.block_all()
        torch.cuda.synchronize()
        self.end_time = time.time()
        if self.log:
            print(f"Elapsed time for {self.name}: {self.elapsed_time:.2f} s")


def get_tensor_memory(tensor, human_readable=True):
    size = tensor.element_size() * tensor.nelement()
    if human_readable:
        size = format_numel_str(size)
    return size


class FeatureSaver:
    def __init__(self, save_dir, bin_size=10, start_bin=0):
        self.save_dir = save_dir
        self.bin_size = bin_size
        self.bin_cnt = start_bin

        self.data_list = []
        self.cnt = 0

    def update(self, data):
        self.data_list.append(data)
        self.cnt += 1

        if self.cnt % self.bin_size == 0:
            self.save()

    def save(self):
        save_path = os.path.join(self.save_dir, f"{self.bin_cnt:08}.bin")
        torch.save(self.data_list, save_path)
        get_logger().info("Saved to %s", save_path)
        self.data_list = []
        self.bin_cnt += 1

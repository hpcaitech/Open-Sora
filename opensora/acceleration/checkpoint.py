import warnings
from collections.abc import Iterable
from typing import Callable, ContextManager, Optional, Tuple

import torch
import torch.nn as nn
from colossalai.utils import get_current_device
from torch.utils.checkpoint import (
    _DEFAULT_DETERMINISM_MODE,
    CheckpointFunction,
    _checkpoint_without_reentrant_generator,
    checkpoint_sequential,
    noop_context_fn,
)


class ActivationManager:
    def __init__(self):
        self.enable = False
        self.buffer = None
        self.total_size = 0
        self.avail_offset = 0
        self.tensor_id_queue = []
        self.ignore_tensor_id_set = set()

    def setup_buffer(self, numel: int, dtype: torch.dtype):
        self.buffer = torch.empty(numel, dtype=dtype, pin_memory=True)
        self.total_size = numel
        self.enable = True

    def offload(self, x: torch.Tensor) -> None:
        if not self.enable or id(x) in self.ignore_tensor_id_set:
            return
        size = x.numel()
        if self.avail_offset + size > self.total_size:
            raise RuntimeError("Activation buffer is full")
        assert x.dtype == self.buffer.dtype, f"Wrong dtype of offload tensor"
        cpu_x = self.buffer[self.avail_offset : self.avail_offset + size].view_as(x)
        cpu_x.copy_(x)
        x.data = cpu_x
        self.avail_offset += size
        self.tensor_id_queue.append(id(x))

    def onload(self, x: torch.Tensor) -> None:
        if not self.enable or id(x) in self.ignore_tensor_id_set:
            return
        assert self.tensor_id_queue[-1] == id(x), f"Wrong order of offload/onload"
        # current x is pinned memory
        assert x.data.is_pinned()
        x.data = x.data.to(get_current_device(), non_blocking=True)
        self.tensor_id_queue.pop()
        self.avail_offset -= x.numel()
        if len(self.tensor_id_queue) == 0:
            self.ignore_tensor_id_set.clear()

    def add_ignore_tensor(self, x: torch.Tensor) -> None:
        self.ignore_tensor_id_set.add(id(x))

    def is_top_tensor(self, x: torch.Tensor) -> bool:
        return len(self.tensor_id_queue) > 0 and self.tensor_id_queue[-1] == id(x)


GLOBAL_ACTIVATION_MANAGER = ActivationManager()


class CheckpointFunctionWithOffload(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        for x in args[::-1]:
            # handle those tensors are used in multiple checkpoints
            if GLOBAL_ACTIVATION_MANAGER.is_top_tensor(x):
                GLOBAL_ACTIVATION_MANAGER.onload(x)
                GLOBAL_ACTIVATION_MANAGER.add_ignore_tensor(x)
        out = CheckpointFunction.forward(ctx, run_function, preserve_rng_state, *args)
        for x in args:
            if torch.is_tensor(x):
                GLOBAL_ACTIVATION_MANAGER.offload(x)
        return out

    @staticmethod
    def backward(ctx, *args):
        # with stack-fashion, the last tensor is the first to be loaded
        for tensor in ctx.saved_tensors[::-1]:
            GLOBAL_ACTIVATION_MANAGER.onload(tensor)
        return CheckpointFunction.backward(ctx, *args)


# TorchDynamo does not step inside utils.checkpoint function.  The flow
# looks likes this
#  1) TorchDynamo tries to wrap utils.checkpoint in a HigherOrderOp by
#     speculatively checking if the forward function is safe to trace.
#  2) If yes, then Dynamo-generated Fx graph has the wrapped higher
#     order op. As a result, TorchDynamo does not look inside utils.checkpoint.
#  3) If not, then TorchDynamo falls back to eager by performing a graph
#     break. And here, the following disable wrapper ensures that
#     TorchDynamo does not trigger again on the frames created by
#     utils.checkpoint innards.
@torch._disable_dynamo
def checkpoint(
    function,
    *args,
    use_reentrant: Optional[bool] = None,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    **kwargs,
):
    r"""Checkpoint a model or part of the model.

    Activation checkpointing is a technique that trades compute for memory.
    Instead of keeping tensors needed for backward alive until they are used in
    gradient computation during backward, forward computation in checkpointed
    regions omits saving tensors for backward and recomputes them during the
    backward pass. Activation checkpointing can be applied to any part of a
    model.

    There are currently two checkpointing implementations available, determined
    by the :attr:`use_reentrant` parameter. It is recommended that you use
    ``use_reentrant=False``. Please refer the note below for a discussion of
    their differences.

    .. warning::

        If the :attr:`function` invocation during the backward pass differs
        from the forward pass, e.g., due to a global variable, the checkpointed
        version may not be equivalent, potentially causing an
        error being raised or leading to silently incorrect gradients.

    .. warning::

        The ``use_reentrant`` parameter should be passed explicitly. In version
        2.4 we will raise an exception if ``use_reentrant`` is not passed.
        If you are using the ``use_reentrant=True`` variant, please refer to the
        note below for important considerations and potential limitations.

    .. note::

        The reentrant variant of checkpoint (``use_reentrant=True``) and
        the non-reentrant variant of checkpoint (``use_reentrant=False``)
        differ in the following ways:

        * Non-reentrant checkpoint stops recomputation as soon as all needed
          intermediate activations have been recomputed. This feature is enabled
          by default, but can be disabled with :func:`set_checkpoint_early_stop`.
          Reentrant checkpoint always recomputes :attr:`function` in its
          entirety during the backward pass.

        * The reentrant variant does not record the autograd graph during the
          forward pass, as it runs with the forward pass under
          :func:`torch.no_grad`. The non-reentrant version does record the
          autograd graph, allowing one to perform backward on the graph within
          checkpointed regions.

        * The reentrant checkpoint only supports the
          :func:`torch.autograd.backward` API for the backward pass without its
          `inputs` argument, while the non-reentrant version supports all ways
          of performing the backward pass.

        * At least one input and output must have ``requires_grad=True`` for the
          reentrant variant. If this condition is unmet, the checkpointed part
          of the model will not have gradients. The non-reentrant version does
          not have this requirement.

        * The reentrant version does not consider tensors in nested structures
          (e.g., custom objects, lists, dicts, etc) as participating in
          autograd, while the non-reentrant version does.

        * The reentrant checkpoint does not support checkpointed regions with
          detached tensors from the computational graph, whereas the
          non-reentrant version does. For the reentrant variant, if the
          checkpointed segment contains tensors detached using ``detach()`` or
          with :func:`torch.no_grad`, the backward pass will raise an error.
          This is because ``checkpoint`` makes all the outputs require gradients
          and this causes issues when a tensor is defined to have no gradient in
          the model. To avoid this, detach the tensors outside of the
          ``checkpoint`` function.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint. Note that under torch.compile,
            this flag doesn't take effect and we always preserve RNG state.
            Default: ``True``
        use_reentrant(bool):
            specify whether to use the activation checkpoint variant that
            requires reentrant autograd. This parameter should be passed
            explicitly. In version 2.4 we will raise an exception if
            ``use_reentrant`` is not passed. If ``use_reentrant=False``,
            ``checkpoint`` will use an implementation that does not require
            reentrant autograd. This allows ``checkpoint`` to support additional
            functionality, such as working as expected with
            ``torch.autograd.grad`` and support for keyword arguments input into
            the checkpointed function.
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
            This argument is only supported if ``use_reentrant=False``.
        determinism_check(str, optional): A string specifying the determinism
            check to perform. By default it is set to ``"default"`` which
            compares the shapes, dtypes, and devices of the recomputed tensors
            against those the saved tensors. To turn off this check, specify
            ``"none"``. Currently these are the only two supported values.
            Please open an issue if you would like to see more determinism
            checks. This argument is only supported if ``use_reentrant=False``,
            if ``use_reentrant=True``, the determinism check is always disabled.
        debug(bool, optional): If ``True``, error messages will also include
            a trace of the operators ran during the original forward computation
            as well as the recomputation. This argument is only supported if
            ``use_reentrant=False``.
        args: tuple containing inputs to the :attr:`function`

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    if use_reentrant is None:
        warnings.warn(
            "torch.utils.checkpoint: the use_reentrant parameter should be "
            "passed explicitly. In version 2.4 we will raise an exception "
            "if use_reentrant is not passed. use_reentrant=False is "
            "recommended, but if you need to preserve the current default "
            "behavior, you can pass use_reentrant=True. Refer to docs for more "
            "details on the differences between the two variants.",
            stacklevel=2,
        )
        use_reentrant = True

    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs and use_reentrant:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    if use_reentrant:
        if context_fn is not noop_context_fn or debug is not False:
            raise ValueError("Passing `context_fn` or `debug` is only supported when " "use_reentrant=False.")
        return CheckpointFunctionWithOffload.apply(function, preserve, *args)
    else:
        gen = _checkpoint_without_reentrant_generator(
            function, preserve, context_fn, determinism_check, debug, *args, **kwargs
        )
        # Runs pre-forward logic
        next(gen)
        ret = function(*args, **kwargs)
        # Runs post-forward logic
        try:
            next(gen)
        except StopIteration:
            return ret


def set_grad_checkpoint(model, use_fp32_attention=False, gc_step=1):
    assert isinstance(model, nn.Module)

    def set_attr(module):
        module.grad_checkpointing = True
        module.fp32_attention = use_fp32_attention
        module.grad_checkpointing_step = gc_step

    model.apply(set_attr)


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, "grad_checkpointing", False):
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, use_reentrant=True, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module, gc_step, *args, use_reentrant=False, **kwargs)
    return module(*args, **kwargs)

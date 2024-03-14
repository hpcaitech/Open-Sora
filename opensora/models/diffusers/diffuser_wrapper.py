"""
Adapted from mmagic
"""

# Copyright (c) OpenMMLab. All rights reserved.
import os
from logging import WARNING
from typing import Any, List, Optional, Union

import torch
from mmengine import print_log
from mmengine.model import BaseModule
from torch import dtype as TORCH_DTYPE

dtype_mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "half": torch.float16,
}


class DiffusersWrapper(BaseModule):
    """Wrapper for models from HuggingFace Diffusers. This wrapper will be set
    a attribute called `_module_cls` by wrapping function and will be used to
    initialize the model structure.

    Example:
    >>> 1. Load pretrained model from HuggingFace Space.
    >>> config = dict(
    >>>     type='ControlNetModel',  # has been registered in `MODELS`
    >>>     from_pretrained='lllyasviel/sd-controlnet-canny',
    >>>     torch_dtype=torch.float16)
    >>> controlnet = MODELS.build(config)

    >>> 2. Initialize model with pre-defined configs.
    >>> config = dict(
    >>>     type='ControlNetModel',  # has been registered in `MODELS`
    >>>     from_config='lllyasviel/sd-controlnet-canny',
    >>>     cache_dir='~/.cache/OpenMMLab')
    >>> controlnet = MODELS.build(config)

    >>> 3. Initialize model with own defined arguments
    >>> config = dict(
    >>>     type='ControlNetModel',  # has been registered in `MODELS`
    >>>     in_channels=3,
    >>>     down_block_types=['DownBlock2D'],
    >>>     block_out_channels=(32, ),
    >>>     conditioning_embedding_out_channels=(16, ))
    >>> controlnet = MODELS.build(config)

    Args:
        from_pretrained (Union[str, os.PathLike], optional): The *model id*
            of a pretrained model or a path to a *directory* containing
            model weights and config. Please refers to
            `diffusers.model.modeling_utils.ModelMixin.from_pretrained`
            for more detail. Defaults to None.
        from_config (Union[str, os.PathLike], optional): The *model id*
            of a pretrained model or a path to a *directory* containing
            model weights and config. Please refers to
            `diffusers.configuration_utils.ConfigMixin.load_config`
            for more detail. Defaults to None.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Noted that, in `DiffuserWrapper`, if you want to load pretrained
            weight from HuggingFace space, please use `from_pretrained`
            argument instead of using `init_cfg`. Defaults to None.

        *args, **kwargs: If `from_pretrained` is passed, *args and **kwargs
            will be passed to `from_pretrained` function. If `from_config`
            is passed, *args and **kwargs will be passed to `load_config`
            function.  Otherwise, *args and **kwargs will be used to
            initialize the model by `self._module_cls(*args, **kwargs)`.
    """

    def __init__(
        self,
        from_pretrained: Optional[Union[str, os.PathLike]] = None,
        from_config: Optional[Union[str, os.PathLike]] = None,
        dtype: Optional[Union[str, TORCH_DTYPE]] = None,
        init_cfg: Union[dict, List[dict], None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(init_cfg)

        module_cls = self._module_cls
        assert not (from_pretrained and from_config), (
            "'from_pretrained' and 'from_config' should not be passed " "at the same time."
        )

        self._from_pretrained = from_pretrained
        self._from_config = from_config

        if from_pretrained is not None:
            self.model = module_cls.from_pretrained(from_pretrained, *args, **kwargs)
            # weight has been initialized from pretrained, therefore we
            # `self._is_init` as True manually;
            # this will cause: mmengine - WARNING - init_weights of DFSAutoencoderKL has been called more than once.
            # Could ignore this warning
            self._is_init = True
        elif from_config is not None:
            _config = module_cls.load_config(from_config, *args, **kwargs)
            self.model = module_cls(**_config)
        else:
            self.model = module_cls(*args, **kwargs)

        if dtype is not None:
            if isinstance(dtype, str):
                assert dtype in dtype_mapping, (
                    "Only support following dtype string: " f"{list(dtype_mapping.keys())}, but receive {dtype}."
                )
                dtype = dtype_mapping[dtype]
            self.model.to(dtype)
            print_log(f"Set model dtype to '{dtype}'.", "current")

        self.config = self.model.config

    def init_weights(self):
        """Initialize the weights.

        If type is 'Pretrained' but the model has be loaded from `repo_id`, a
        warning will be raised.
        """
        if self.init_cfg and self.init_cfg["type"] == "Pretrained":
            if self._from_pretrained is not None:
                print_log(
                    "Has been loaded from pretrained model from "
                    f"'{self._from_pretrained}'. Your behavior is "
                    "very dangerous.",
                    "current",
                    WARNING,
                )
        super().init_weights()

    def __getattr__(self, name: str) -> Any:
        """This function provide a way to access the attributes of the wrapped
        model.

        Args:
            name (str): The name of the attribute.

        Returns:
            Any: The got attribute.
        """
        # Q: why we need end of recursion for 'model'?
        # A: In `nn.Module.__setattr__`, if value is instance of `nn.Module`,
        #   it will be removed from `__dict__` and added to
        #   `__dict__._modules`. Therefore, `model` cannot be found in
        #   `self.__dict__`. When we call `self.model`, python cannot found
        #   'model' in `self.__dict__` and then `self.__getattr__('model')`
        #   will be called. If we call `self.model` in `self.__getattr__`
        #   which does not have any exit about 'model',`RecursionError`
        #   will be raised.
        if name == "model":
            return super().__getattr__("model")

        try:
            return getattr(self.model, name)
        except AttributeError:
            try:
                return super().__getattr__(name)
            except AttributeError:
                raise AttributeError(
                    "'name' cannot be found in both "
                    f"'{self.__class__.__name__}' and "
                    f"'{self.__class__.__name__}.model'."
                )

    def __repr__(self):
        """The representation of the wrapper."""
        s = super().__repr__()
        prefix = f"Wrapped Module Class: {self._module_cls}\n"
        prefix += f"Wrapped Module Name: {self._module_name}\n"
        if self._from_pretrained:
            prefix += f"From Pretrained: {self._from_pretrained}\n"
        if self._from_config:
            prefix += f"From Config: {self._from_config}\n"
        s = prefix + s
        return s

    def forward(self, *args, **kwargs) -> Any:
        """Forward function of wrapped module.

        Args:
            *args, **kwargs: The arguments of the wrapped module.

        Returns:
            Any: The output of wrapped module's forward function.
        """
        return self.model(*args, **kwargs)

    def to(
        self,
        torch_device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """Put wrapped module to device or convert it to torch_dtype. There are
        two to() function. One is nn.module.to() and the other is
        diffusers.pipeline.to(), if both args are passed,
        diffusers.pipeline.to() is called.

        Args:
            torch_device: The device to put to.
            torch_dtype: The type to convert to.

        Returns:
            self: the wrapped module itself.
        """
        if torch_dtype is None:
            self.model.to(torch_device)
        else:
            self.model.to(torch_device, torch_dtype)
        return self

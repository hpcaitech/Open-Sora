"""
Adapted from mmagic
"""

import inspect
import warnings

from opensora.registry import MODELS
from opensora.utils.misc import try_import

from .diffuser_wrapper import DiffusersWrapper


def register_diffusers_models(model_list):
    """Register models in ``diffusers.models`` to the ``MODELS`` registry.
    Specifically, the registered models from diffusers only defines the network
    forward without training. See more details about diffusers in:
    https://huggingface.co/docs/diffusers/api/models.

    Returns:
        List[str]: A list of registered DIFFUSION_MODELS' name.
    """

    diffusers = try_import("diffusers")
    if diffusers is None:
        warnings.warn(
            "Diffusion Models are not registered as expect. "
            "If you want to use diffusion models, "
            "please install diffusers>=0.12.0."
        )
        return None

    def gen_wrapped_cls(module, module_name):
        return type(
            module_name, (DiffusersWrapper,), dict(_module_cls=module, _module_name=module_name, __module__=__name__)
        )

    DIFFUSERS_MODELS = []
    # for module_name in dir(diffusers.models):
    for module_name in model_list:
        module = getattr(diffusers.models, module_name)
        if inspect.isclass(module):
            register_name = f"DFS{module_name}"  # DFS for diffusers
            wrapped_module = gen_wrapped_cls(module, register_name)
            MODELS.register_module(name=register_name, module=wrapped_module)

            DIFFUSERS_MODELS.append(register_name)

    return DIFFUSERS_MODELS


REGISTERED_DIFFUSERS_MODELS = register_diffusers_models(["AutoencoderKL", "AutoencoderKLTemporalDecoder"])

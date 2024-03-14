from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import Registry

#######################################################################
# opensora.datasets
#######################################################################

DATASETS = Registry(
    "dataset",
    parent=MMENGINE_DATASETS,
    locations=[
        "opensora.datasets"
    ],  # triggered in DATASETS.build() => build_from_cfg() => Registry.get() => Registry.import_from_location()
)

# TRANSFORMS = Registry(
#     'transform',
#     parent=MMENGINE_TRANSFORMS,
#     locations=['opensora.datasets.transforms'],
# )

# DATA_SAMPLERS = Registry(
#     'data sampler',
#     parent=MMENGINE_DATA_SAMPLERS,
#     locations=['opensora.datasets'],
# )

# FUNCTIONS = Registry(
#     'function',
#     parent=MMENGINE_FUNCTIONS,
#     locations=['opensora.datasets'],
# )

#######################################################################
# opensora.models
#######################################################################

MODELS = Registry(
    "model",
    parent=MMENGINE_MODELS,
    locations=["opensora.models"],
)

# MODEL_WRAPPERS = Registry(
#     'model_wrapper',
#     parent=MMENGINE_MODEL_WRAPPERS,
#     locations=['opensora.models.wrappers'],
# )

DIFFUSION_SCHEDULERS = Registry(
    "diffusion scheduler",
    locations=["opensora.diffusion"],
)

#######################################################################
# opensora.engine
#######################################################################

# RUNNERS = Registry(
#     'runner',
#     parent=MMENGINE_RUNNERS,
#     locations=['opensora.engine.runners'],
# )

# LOOPS = Registry(
#     'loop',
#     parent=MMENGINE_LOOPS,
#     locations=['opensora.engine.loops'],
# )

# HOOKS = Registry(
#     'hook',
#     parent=MMENGINE_HOOKS,
#     locations=['opensora.hooks'],
# )

# OPTIM_WRAPPER_CONSTRUCTORS = Registry(
#     'optimizer wrapper constructor',
#     parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
#     locations=['opensora.engine.optimizers'],
# )
#
# OPTIM_WRAPPERS = Registry(
#     'optim_wrapper',
#     parent=MMENGINE_OPTIM_WRAPPERS,
#     locations=['opensora.engine.optimizers'],
# )
#
# OPTIMIZERS = Registry(
#     'optimizer',
#     parent=MMENGINE_OPTIMIZERS,
#     locations=['opensora.engine.optimizers'],
# )

# METRICS = Registry(
#     'metric',
#     parent=MMENGINE_METRICS,
#     locations=['opensora.evaluation'],
# )
#
# VISBACKENDS = Registry(
#     'vis_backend',
#     parent=MMVISBACKENDS,
#     locations=['opensora.visualization'],
# )

# LOG_PROCESSORS = Registry(
#     'log_processor',
#     parent=MMLOG_PROCESSORS,
#     locations=['opensora.engine.runners'],
# )

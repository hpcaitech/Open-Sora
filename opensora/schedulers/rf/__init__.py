# should have property num_timesteps, 
# method sample()  training_losses()
import torch
from .rectified_flow import RFlowScheduler
from functools import partial

from opensora.registry import SCHEDULERS

# @SCHEDULERS.register_module("rflow")
# class RFLOW:
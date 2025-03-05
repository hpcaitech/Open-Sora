import torch
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from torch.optim.lr_scheduler import _LRScheduler


def create_optimizer(
    model: torch.nn.Module,
    optimizer_config: dict,
) -> torch.optim.Optimizer:
    """
    Create an optimizer.

    Args:
        model (torch.nn.Module): The model to be optimized.
        optimizer_config (dict): The configuration of the optimizer.

    Returns:
        torch.optim.Optimizer: The optimizer.
    """
    optimizer_name = optimizer_config.pop("cls", "HybridAdam")
    if optimizer_name == "HybridAdam":
        optimizer_cls = HybridAdam
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    optimizer = optimizer_cls(
        filter(lambda p: p.requires_grad, model.parameters()),
        **optimizer_config,
    )
    return optimizer


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_steps_per_epoch: int,
    epochs: int = 1000,
    warmup_steps: int | None = None,
    use_cosine_scheduler: bool = False,
    initial_lr: float = 1e-6,
) -> _LRScheduler | None:
    """
    Create a learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        num_steps_per_epoch (int): The number of steps per epoch.
        epochs (int): The number of epochs.
        warmup_steps (int |  None): The number of warmup steps.
        use_cosine_scheduler (bool): Whether to use cosine scheduler.

    Returns:
        _LRScheduler |  None: The learning rate scheduler
    """
    if warmup_steps is None and not use_cosine_scheduler:
        lr_scheduler = None
    elif use_cosine_scheduler:
        lr_scheduler = CosineAnnealingWarmupLR(
            optimizer,
            total_steps=num_steps_per_epoch * epochs,
            warmup_steps=warmup_steps,
        )
    else:
        lr_scheduler = LinearWarmupLR(optimizer, initial_lr=1e-6, warmup_steps=warmup_steps)
        # lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=warmup_steps)

    return lr_scheduler


class LinearWarmupLR(_LRScheduler):
    """Linearly warmup learning rate and then linearly decay.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0
        last_step (int, optional): The index of last step, defaults to -1. When last_step=-1,
            the schedule is started from the beginning or When last_step=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, initial_lr=0, warmup_steps: int = 0, last_epoch: int = -1):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                self.initial_lr + (self.last_epoch + 1) / (self.warmup_steps + 1) * (lr - self.initial_lr)
                for lr in self.base_lrs
            ]
        else:
            return self.base_lrs

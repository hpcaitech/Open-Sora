import random
from collections import OrderedDict

import torch


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, optimizer=None, decay: float = 0.9999, sharded: bool = True
) -> None:
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == "pos_embed":
            continue
        if param.requires_grad == False:
            continue
        if not sharded:
            param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
        else:
            if param.data.dtype != torch.float32:
                param_id = id(param)
                master_param = optimizer._param_store.working_to_master_param[param_id]
                param_data = master_param.data
            else:
                param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)


class MaskGenerator:
    def __init__(self, mask_ratios):
        self.mask_name = ["mask_no", "mask_random", "mask_head", "mask_tail", "mask_head_tail"]
        self.mask_prob = mask_ratios
        print(self.mask_prob)
        self.mask_acc_prob = [sum(self.mask_prob[: i + 1]) for i in range(len(self.mask_prob))]

    def get_mask(self, x):
        mask_type = random.random()
        for i, acc_prob in enumerate(self.mask_acc_prob):
            if mask_type <= acc_prob:
                mask_name = self.mask_name[i]
                break

        mask = torch.ones(x.shape[2], dtype=torch.bool, device=x.device)
        if mask_name == "mask_random":
            random_size = random.randint(1, 4)
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
            return mask
        elif mask_name == "mask_head":
            random_size = random.randint(1, 4)
            mask[:random_size] = 0
        elif mask_name == "mask_tail":
            random_size = random.randint(1, 4)
            mask[-random_size:] = 0
        elif mask_name == "mask_head_tail":
            random_size = random.randint(1, 4)
            mask[:random_size] = 0
            mask[-random_size:] = 0

        return mask

    def get_masks(self, x):
        masks = []
        for _ in range(len(x)):
            mask = self.get_mask(x)
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        return masks

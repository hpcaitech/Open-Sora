from datetime import timedelta

from opensora.utils.misc import to_torch_dtype
import torch
import torch_musa
import pandas as pd
import torch.distributed as dist

import colossalai
from tqdm import tqdm
from collections import OrderedDict
from colossalai.cluster import DistCoordinator
from colossalai.utils import get_current_device, set_seed

from opensora.registry import MODELS, build_module
from opensora.utils.config_utils import (
    create_experiment_workspace,
    parse_configs,
    save_training_config,
)


def main():
    # ======================================================
    # 1. args & cfg
    # ======================================================
    cfg = parse_configs(training=True)
    exp_name, exp_dir = create_experiment_workspace(cfg)
    save_training_config(cfg._cfg_dict, exp_dir)

    # ======================================================
    # 2. runtime variables & colossalai launch
    # ======================================================
    assert torch.musa.is_available(), "Training currently requires at least one GPU."
    assert cfg.dtype in ["fp32", "fp16", "bf16"], f"Unknown mixed precision {cfg.dtype}"
    # 2.1. colossalai init distributed training
    dist.init_process_group(backend="mccl", timeout=timedelta(hours=24))
    torch.musa.set_device(dist.get_rank() % torch.musa.device_count())
    set_seed(1024)
    coordinator = DistCoordinator()
    device = get_current_device()  # device musa:0
    dtype = to_torch_dtype(cfg.dtype)

    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    data_path = cfg.dataset['data_path']
    dataset = pd.read_csv(data_path)
    # path to save tensor
    csv_save_path = ''.join(data_path.split('.')[0] + '_text_idx' + '.' + data_path.split('.')[1])
    model_args_save_path = ''.join(data_path.split('.')[0] + '_model_args' + '.' + 'pt')
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    text_idx_list = []
    text_idx_to_dict = OrderedDict() # {id: model_args {'y': ..., 'mask':...}}
    for idx, item in dataset.iterrows():
        video_path = item['path']
        text = item['text']
        with torch.no_grad():
            model_args = text_encoder.forward(text)
            # print(f"model_args {model_args}")
            model_args['y'] = model_args['y'].to(device='cpu', dtype=model_args['y'].dtype)
            model_args['mask'] = model_args['mask'].to(device='cpu')
            text_id = id(video_path)
            text_idx_list.append(text_id)
            text_idx_to_dict[text_id] = model_args
    dataset['text_idx'] = text_idx_list
    
    # save updated csv
    dataset.to_csv(csv_save_path, index=False)
    # save id:model_args
    torch.save(text_idx_to_dict, model_args_save_path)
    
    # check load
    model_args = torch.load(model_args_save_path)

# Run with:
# torchrun --nnodes=1 --nproc_per_node=1 scripts/preprocess_text.py configs/opensora/train/16x256x256.py --data-path /home/dist/hpcai/duanjunwen/Open-Sora/dataset/panda3m/meta/meta_clips_caption_cleaned.csv
if __name__ == "__main__":
    main()

from copy import deepcopy
from datetime import timedelta, datetime
from pprint import pprint
# import thop

import torch
import torch_musa
import pandas as pd
import thop
import time
from calflops import calculate_flops
# import numpy as np
import torch.distributed as dist
# import wandb
import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.booster.plugin import TorchFSDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from wonderwords import RandomWord, RandomSentence
# from torch.optim import Adam, AdamW
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm
import functools
from functools import partial

from performance_evaluator import PerformanceEvaluator
from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)
from opensora.acceleration.plugin import ZeroSeqParallelPlugin
from opensora.datasets import prepare_dataloader, prepare_variable_dataloader
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import create_logger, load, model_sharding, record_model_param_shape, save
from opensora.utils.config_utils import (
    create_experiment_workspace,
    create_tensorboard_writer,
    parse_configs,
    save_training_config,
)
from opensora.utils.misc import all_reduce_mean, format_numel_str,format_numel, get_model_numel, requires_grad, to_torch_dtype
from opensora.utils.train_utils import MaskGenerator, update_ema
from opensora.models.stdit.stdit import STDiT, STDiTBlock
from opensora.models.stdit.stdit2 import STDiT2, STDiT2Block

def register_hooks(module):
    def bwd_hook(module, grad_input, grad_output):
        torch.musa.synchronize()
        runtime = time.time() - module.backward_start_time
        print(f"stdit {module._name} bwd runtime {runtime * 1000} ms")
        # if 'proj' in module._name:
        #     print(f"{module._name} {module.dtype}")
        # print(f"{module._name} post hook\n")
        # print(f"Grad input {grad_input} type: {type(grad_input)}; len: {len(grad_input)}; shape {grad_input[0].shape}\n")
        # print(f"Grad output {grad_output} type: {type(grad_output)}; len: {len(grad_output)}; shape {grad_output[0].shape}\n")
        # print(f"Grad input type: {type(grad_input)}; len: {len(grad_input)}; shape {grad_input[0].shape}\n")
        # print(f"Grad output type: {type(grad_output)}; len: {len(grad_output)}; shape {grad_output[0].shape}\n")

    def bwd_pre_hook(module, grad_output):
        torch.musa.synchronize()
        module.backward_start_time=time.time()
        # print(f"{module._name} pre hook\n")
        # print(f"Grad output {grad_output} type: {type(grad_output)}; len: {len(grad_output)}; shape {grad_output[0].shape}\n")
        # print(f"Grad output type: {type(grad_output)}; len: {len(grad_output)}; shape {grad_output[0].shape}\n")
    
    module.register_full_backward_pre_hook(bwd_pre_hook)
    module.register_backward_hook(bwd_hook)
    # module.register_full_backward_pre_hook(bwd_pre_hook)
    
    
class ProfileModule(torch.nn.Module):
	def __init__(self, module, fn='encode'):
		super().__init__()
		self.module = module
		self.forward_func = getattr(module, fn)

	def forward(self, *args):
		return self.forward_func(*args)

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
    # assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert torch.musa.is_available(), "Training currently requires at least one GPU."
    assert cfg.dtype in ["fp32", "fp16", "bf16"], f"Unknown mixed precision {cfg.dtype}"
    # torch.backends.cuda.enable_flash_sdp(enabled=False) # MUSA only support flash atten dim <= 128; but pretrained has 512
    # 2.1. colossalai init distributed training
    # we set a very large timeout to avoid some processes exit early
    # dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    dist.init_process_group(backend="mccl", timeout=timedelta(hours=24))
    # torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    torch.musa.set_device(dist.get_rank() % torch.musa.device_count())
    set_seed(1024)
    coordinator = DistCoordinator()
    device = get_current_device()  # device musa:0
    dtype = to_torch_dtype(cfg.dtype)

    # 2.2. init logger, tensorboard & wandb
    if not coordinator.is_master():
        logger = create_logger(None)
    else:
        print("Training configuration:")
        pprint(cfg._cfg_dict)
        logger = create_logger(exp_dir)
        logger.info(f"Experiment directory created at {exp_dir}")

        writer = create_tensorboard_writer(exp_dir)
        if cfg.wandb:
            wandb.init(project="minisora", name=exp_name, config=cfg._cfg_dict)

    # 2.3. initialize ColossalAI booster
    if cfg.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
        )
        set_data_parallel_group(dist.group.WORLD)
    elif cfg.plugin == "zero2-seq":
        plugin = ZeroSeqParallelPlugin(
            sp_size=cfg.sp_size,
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
        )
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
    elif cfg.plugin == "fsdp":
        plugin = TorchFSDPPlugin()
        set_data_parallel_group(dist.group.WORLD)
    elif cfg.plugin == "ddp":
        plugin = TorchDDPPlugin()
        set_data_parallel_group(dist.group.WORLD)
    else:
        raise ValueError(f"Unknown plugin {cfg.plugin}")
    booster = Booster(plugin=plugin)
    
    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info(f"Dataset contains {len(dataset)} samples.")
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )
    # TODO: use plugin's prepare dataloader
    if cfg.bucket_config is None:
        dataloader = prepare_dataloader(**dataloader_args)
    else:
        dataloader = prepare_variable_dataloader(
            bucket_config=cfg.bucket_config,
            num_bucket_build_workers=cfg.num_bucket_build_workers,
            **dataloader_args,
        )
    if cfg.dataset.type == "VideoTextDataset":
        total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.sp_size
        logger.info(f"Total batch size: {total_batch_size}")

    # ======================================================
    # 4. build model
    # ======================================================
    # 4.1. build model
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS)
    input_size = (dataset.num_frames, *dataset.image_size)
    latent_size = vae.get_latent_size(input_size)
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        dtype=dtype,
    )
    model_numel, model_numel_trainable = get_model_numel(model)
    t5_numel, t5_numel_trainable = get_model_numel(text_encoder)
    vae_numel, vae_numel_trainable = get_model_numel(vae)

    logger.info(
        f"Trainable model params: {format_numel_str(model_numel_trainable)}, Total model params: {format_numel_str(model_numel)}"
    )

    # 4.2. create ema
    ema = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)

    # 4.3. move to device
    vae = vae.to(device, dtype)
    model = model.to(device, dtype)

    # 4.4. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # 4.5. setup optimizer
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=0,
        adamw_mode=True,
    )
    
    # optimizer = Adam(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=cfg.lr,
    #     weight_decay=0,
    # )
    
    lr_scheduler = None

    # 4.6. prepare for training
    if cfg.grad_checkpoint:
        set_grad_checkpoint(model, (STDiTBlock, STDiT2Block))
        num_ckpt_blocks = 0
        for module in model.modules():
            if isinstance(module, (STDiTBlock, STDiT2Block)):
                module.grad_checkpointing = module.grad_checkpointing and num_ckpt_blocks < cfg.num_ckpt_blocks
                num_ckpt_blocks += module.grad_checkpointing
    model.train()
    update_ema(ema, model, decay=0, sharded=False)
    ema.eval()
    if cfg.mask_ratios is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)
    
    # 4.7. initialize Evaluator
    Evaluator = functools.partial(
        PerformanceEvaluator,
        model_numel=model_numel,
        num_layers=model.num_heads,
        hidden_size=model.hidden_size,
        vocab_size=text_encoder.output_dim,
        max_seq_length=512,
        ignore_steps=2,
        num_steps=cfg.benchmark_num_steps, # epoch * steps 
        use_torch_profiler=False,
        cfg=cfg,
        # num_steps=22, # epoch * steps 
        # use_torch_profiler=True,
        # torch_profiler_path=f"./profiler/{plugin}",
        # ignore_steps=2,
    )
    
    # =======================================================
    # 5. boost model for distributed training with colossalai
    # =======================================================
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boost model for distributed training")
    if cfg.dataset.type == "VariableVideoTextDataset":
        num_steps_per_epoch = dataloader.batch_sampler.get_num_batch() // dist.get_world_size()
    else:
        num_steps_per_epoch = len(dataloader)
    performance_evaluator = Evaluator(stdit_weight_memory=format_numel(model_numel*2),total_weight_memory=format_numel((model_numel+t5_numel+vae_numel)*2))
    s = RandomSentence()
    # # =======================================================
    # # 6. test flops; OOM in vae calculate_flops;
    # # =======================================================
    # print(f"Memory Usage After boost {torch.musa.memory_allocated()/1024**3:.3f} GB,")
    # x = torch.randn(cfg.batch_size, 3, cfg.dataset["num_frames"], cfg.dataset["image_size"][0], cfg.dataset["image_size"][1], dtype=dtype).to(device)  # (B, C, T, H, W) 
    # y = [s.simple_sentence() for _ in range(cfg.batch_size)] 
    # print(f"Memory Usage After generate data {torch.musa.memory_allocated()/1024**3:.3f} GB,")
    # vae_encode = ProfileModule(module=vae, fn='encode')
    # print(f"Memory Usage After Profile vae {torch.musa.memory_allocated()/1024**3:.3f} GB,")
    # vae_flops, vae_macs, vae_params = calculate_flops(model=vae_encode,args=[x])
    # performance_evaluator.vae_flop = vae_flops
    # print(f"vae flops {vae_flops}, vae_macs {vae_macs}, vae_params {vae_params}")
    
    # with torch.no_grad():
    #     # Prepare visual inputs
    #     x = vae.encode(x)  # [B, C, T, H/P, W/P]
    #     # Prepare text inputs
    #     model_args = text_encoder.forward(y)
    # # Mask
    # if cfg.mask_ratios is not None:
    #     mask = mask_generator.get_masks(x)
    #     model_args["x_mask"] = mask
    # else:
    #     mask = None
    # t = torch.randint(0, scheduler.num_timesteps, (x.shape[0],), device=device)
    # if cfg.model["type"] == "STDiT-XL/2":
    #     stdit_flops, stdit_macs, stdit_params = calculate_flops(model=model,args=[x, t ,model_args['y'], model_args['mask']], include_backPropagation=True)
    # elif cfg.model["type"] == "STDiT2-XL/2":
    #     model_args['num_frames'] = torch.randn(cfg.dataset["num_frames"], dtype=dtype).to(device) 
    #     model_args['height'] = torch.randn(cfg.batch_size, dtype=dtype).to(device) 
    #     model_args['width'] = torch.randn(cfg.batch_size, dtype=dtype).to(device)
    #     model_args['ar'] = torch.randn(cfg.batch_size, dtype=dtype).to(device) 
    #     model_args['fps'] = torch.randn(cfg.batch_size, dtype=dtype).to(device) 
    #     stdit_flops, stdit_macs, stdit_params = calculate_flops(model=model,args=[x, t , model_args['y'], model_args['mask'], model_args["x_mask"], model_args['num_frames'], model_args['height'], model_args['width'], model_args['ar'], model_args['fps']], include_backPropagation=True)
    # print(f"stdit_flops {stdit_flops} macs {stdit_macs} params {stdit_params}")
    # performance_evaluator.stdit_flop = stdit_flops
          
    # =======================================================
    # 7. training loop
    # =======================================================

    start_epoch = start_step = log_step = sampler_start_idx = acc_step = 0
    running_loss = 0.0
    sampler_to_io = dataloader.batch_sampler if cfg.dataset.type == "VariableVideoTextDataset" else None
    # 6.1. resume training
    if cfg.load is not None:
        logger.info("Loading checkpoint")
        ret = load(
            booster,
            model,
            ema,
            optimizer,
            lr_scheduler,
            cfg.load,
            sampler=sampler_to_io if not cfg.start_from_scratch else None,
        )
        if not cfg.start_from_scratch:
            start_epoch, start_step, sampler_start_idx = ret
        logger.info(f"Loaded checkpoint {cfg.load} at epoch {start_epoch} step {start_step}")
    logger.info(f"Training for {cfg.epochs} epochs with {num_steps_per_epoch} steps per epoch")

    if cfg.dataset.type == "VideoTextDataset":
        dataloader.sampler.set_start_index(sampler_start_idx)
    model_sharding(ema)
    # 6.2. training loop
    performance_evaluator.on_fit_start()
    for name, module in model.named_modules(prefix="stdit"):
        module._name = name
    
    loss_list = list()
    # model.apply(register_hooks)
    if cfg.random_dataset:
        num_steps_per_epoch = cfg.benchmark_num_steps
    for epoch in range(start_epoch, cfg.epochs):
        if cfg.dataset.type == "VideoTextDataset":
            dataloader.sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info(f"Beginning epoch {epoch}...")
        performance_evaluator.start_new_iter()
        for step in range(num_steps_per_epoch):
            x = torch.randn(cfg.batch_size, 3, cfg.dataset["num_frames"], cfg.dataset["image_size"][0], cfg.dataset["image_size"][1], dtype=dtype).to(device)  # (B, C, T, H, W) 
            y = [s.simple_sentence() for _ in range(cfg.batch_size)]    
                
            # Visual and text encoding
            with torch.no_grad():
                # Prepare visual inputs
                performance_evaluator.before_video_encode()
                x = vae.encode(x)  # [B, C, T, H/P, W/P]
                # Prepare text inputs
                performance_evaluator.before_text_encode()
                model_args = text_encoder.forward(y)
                         
            # Mask
            if cfg.mask_ratios is not None:
                mask = mask_generator.get_masks(x)
                model_args["x_mask"] = mask
            else:
                mask = None
            # Video info
            # for k, v in batch.items():
            #     if k not in ['video', 'text']:
            #         model_args[k] = v.to(device, dtype)
                    
            # Diffusion
            t = torch.randint(0, scheduler.num_timesteps, (x.shape[0],), device=device)
            if cfg.model["type"] == "STDiT2-XL/2":
                model_args['num_frames'] = torch.randn(cfg.batch_size, dtype=dtype).to(device) 
                model_args['height'] = torch.randn(cfg.batch_size, dtype=dtype).to(device) 
                model_args['width'] = torch.randn(cfg.batch_size, dtype=dtype).to(device)
                model_args['ar'] = torch.randn(cfg.batch_size, dtype=dtype).to(device) 
                model_args['fps'] = torch.randn(cfg.batch_size, dtype=dtype).to(device)
            performance_evaluator.before_forward()
            loss_dict = scheduler.training_losses(model, x, t, model_args, mask=mask)
            # Backward & update
            loss = loss_dict["loss"].mean()
            logger.info(f"loss: {loss}\n")
                
            performance_evaluator.before_backward()
            booster.backward(loss=loss, optimizer=optimizer)
            performance_evaluator.before_optimizer_update()
            optimizer.step()
            optimizer.zero_grad()
                
            # Update EMA
            update_ema(ema, model.module, optimizer=optimizer)
                
            # Log loss values:
            all_reduce_mean(loss)
                
            if coordinator.is_master():
                loss_list.append(float(loss.to('cpu')))
                    
            running_loss += loss.item()
            global_step = epoch * num_steps_per_epoch + step
            log_step += 1
            acc_step += 1

            # Save checkpoint
            if cfg.ckpt_every > 0 and (global_step + 1) % cfg.ckpt_every == 0:
                save(
                        booster,
                        model,
                        ema,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        step + 1,
                        global_step + 1,
                        cfg.batch_size,
                        coordinator,
                        exp_dir,
                        ema_shape_dict,
                        sampler=sampler_to_io,
                    )
                logger.info(
                        f"Saved checkpoint at epoch {epoch} step {step + 1} global_step {global_step + 1} to {exp_dir}"
                )
            performance_evaluator.end_iter(input_ids=torch.empty(cfg.batch_size, cfg.epochs))
            performance_evaluator.start_new_iter()

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        if cfg.dataset.type == "VideoTextDataset":
            dataloader.sampler.set_start_index(0)
        if cfg.dataset.type == "VariableVideoTextDataset":
            dataloader.batch_sampler.set_epoch(epoch + 1)
            print("Epoch done, recomputing batch sampler")
        start_step = 0
        
    if coordinator.is_master():
        df = pd.DataFrame(loss_list)
        df.to_csv(f"./loss_curve/loss_curve_{datetime.now()}.csv",index=False)

    performance_evaluator.on_fit_end()

if __name__ == "__main__":
    main()

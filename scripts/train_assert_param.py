from copy import deepcopy
from datetime import timedelta
from pprint import pprint
# import thop

import random
import numpy as np
import torch
import torch_musa
import torch.distributed as dist
# import wandb
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.booster.plugin import TorchFSDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
# from torch.optim import Adam, AdamW
from opensora.utils.train_utils import set_seed
from colossalai.utils import get_current_device
from tqdm import tqdm
import functools
from functools import partial

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
from opensora.utils.misc import all_reduce_mean, format_numel_str, get_model_numel, requires_grad, to_torch_dtype
from opensora.utils.train_utils import MaskGenerator, update_ema


def register_hooks(module):
    def bwd_hook(module, grad_input, grad_output):
        torch.musa.synchronize()
        # if 'proj' in module._name:
        #     print(f"{module._name} {module.dtype}")
        # print(f"{module._name} post hook\n")
        # print(f"Grad input {grad_input} type: {type(grad_input)}; len: {len(grad_input)}; shape {grad_input[0].shape}\n")
        # print(f"Grad output {grad_output} type: {type(grad_output)}; len: {len(grad_output)}; shape {grad_output[0].shape}\n")
        # print(f"Grad input type: {type(grad_input)}; len: {len(grad_input)}; shape {grad_input[0].shape}\n")
        # print(f"Grad output type: {type(grad_output)}; len: {len(grad_output)}; shape {grad_output[0].shape}\n")
        torch.save(grad_input, f"./dataset/assert_closed/{module._name}_grad_input.txt")
        torch.save(grad_output, f"./dataset/assert_closed/{module._name}_grad_output.txt")

    def bwd_pre_hook(module, grad_output):
        torch.musa.synchronize()
        # print(f"{module._name} pre hook\n")
        # print(f"Grad output {grad_output} type: {type(grad_output)}; len: {len(grad_output)}; shape {grad_output[0].shape}\n")
        # print(f"Grad output type: {type(grad_output)}; len: {len(grad_output)}; shape {grad_output[0].shape}\n")
    
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
    
    lr_scheduler = None

    # 4.6. prepare for training
    if cfg.grad_checkpoint:
        set_grad_checkpoint(model)
    model.train()
    update_ema(ema, model, decay=0, sharded=False)
    ema.eval()
    if cfg.mask_ratios is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)
    
    
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
    # =======================================================
    # 6. training loop
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

    for name, module in model.named_modules(prefix="stdit"):
        module._name = name
    
    
    # TODO: assert vae, text encoder, stdit 
    
    # # T5
    # for name, param in text_encoder.t5.model.named_parameters():
    #     print(f"{name} param shape {param.shape}\n param\n {param}\n") 
    #     break
    
    # Assert 1: save vae init
    torch.save(vae.state_dict(), f"./dataset/assert_closed/musa_tensor/vae_param_init.txt")
    # # load vae
    # vae.load_state_dict(torch.load(f"./dataset/assert_closed/{vae.__class__.__name__}_init.txt"))
    
    # Assert 2: save t5
    # torch.save(text_encoder.state_dict(), f"./dataset/assert_closed/{text_encoder.__class__.__name__}_init.txt")
    
    # Assert 3: save stdit init
    torch.save(model.state_dict(), f"./dataset/assert_closed/musa_tensor/stdit_param_init.txt")
    
    # Assert 4: save optim state
    torch.save(optimizer.state_dict(), f"./dataset/assert_closed/musa_tensor/optim_state_init.txt")
    # optimizer.load_state_dict(torch.load(f"./dataset/assert_closed/{optimizer.__class__.__name__}_init.txt"))
    
    # loss_list = list()
    # model.apply(register_hooks)
    for epoch in range(start_epoch, cfg.epochs):
        if cfg.dataset.type == "VideoTextDataset":
            dataloader.sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info(f"Beginning epoch {epoch}...")

        
        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            for step, batch in pbar:
                # TODO: In each step assert: vae input, vae output, t5 input, t5 output, stdit grad
                x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
                y = batch.pop("text")
                
                # Assert 5: assert vae input
                torch.save(x, f"./dataset/assert_closed/musa_tensor/step{epoch * num_steps_per_epoch + step}_vae_input.txt")
                
                # Assert 6: assert t5 input
                torch.save(y, f"./dataset/assert_closed/musa_tensor/step{epoch * num_steps_per_epoch + step}_t5_input.txt")

                # Visual and text encoding
                with torch.no_grad():
                    # Prepare visual inputs
                    x = vae.encode(x)  # [B, C, T, H/P, W/P]
                    # Prepare text inputs
                    # model_args = text_encoder.encode(y)
                    model_args = text_encoder.forward(y)
                
                # Assert 6: assert vae output
                torch.save(x, f"./dataset/assert_closed/musa_tensor/step{epoch * num_steps_per_epoch + step}_vae_output.txt")
                # Assert 8: assert t5 output
                torch.save(model_args, f"./dataset/assert_closed/musa_tensor/step{epoch * num_steps_per_epoch + step}_t5_output.txt")
                
                
                # Mask
                # print(f"model_args {model_args}")
                if cfg.mask_ratios is not None:
                    mask = mask_generator.get_masks(x)
                    model_args["x_mask"] = mask
                else:
                    mask = None
                # Video info
                for k, v in batch.items():
                    model_args[k] = v.to(device, dtype)
                
                
                # Diffusion
                t = torch.randint(0, scheduler.num_timesteps, (x.shape[0],), device=device)
                loss_dict = scheduler.training_losses(model, x, t, model_args, mask=mask)
                # Backward & update
                loss = loss_dict["loss"].mean()
                # Assert 9: assert loss
                logger.info(f"loss: {loss}\n")
                
                booster.backward(loss=loss, optimizer=optimizer)
                
                # Assert 10: assert stdit grad (in model.apply(register_hooks))

                optimizer.step()
                optimizer.zero_grad()
                
                # Assert 11: assert stdit param
                torch.save(model.state_dict(), f"./dataset/assert_closed/musa_tensor/step{epoch * num_steps_per_epoch + step}_stdit_param_step.txt")
                # Assert 12: assert optim state    
                torch.save(optimizer.state_dict(), f"./dataset/assert_closed/musa_tensor/step{epoch * num_steps_per_epoch + step}_optim_state_step.txt")
                
                # Update EMA
                update_ema(ema, model.module, optimizer=optimizer)
                
                # Log loss values:
                all_reduce_mean(loss)
                running_loss += loss.item()
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1
                acc_step += 1

                # Log to tensorboard
                if coordinator.is_master() and (global_step + 1) % cfg.log_every == 0:
                    avg_loss = running_loss / log_step
                    pbar.set_postfix({"loss": avg_loss, "step": step, "global_step": global_step})
                    running_loss = 0
                    log_step = 0
                    writer.add_scalar("loss", loss.item(), global_step)
                    if cfg.wandb:
                        wandb.log(
                            {
                                "iter": global_step,
                                "epoch": epoch,
                                "loss": loss.item(),
                                "avg_loss": avg_loss,
                                "acc_step": acc_step,
                            },
                            step=global_step,
                        )

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
                
                # assert only one step now 
                break

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        if cfg.dataset.type == "VideoTextDataset":
            dataloader.sampler.set_start_index(0)
        if cfg.dataset.type == "VariableVideoTextDataset":
            dataloader.batch_sampler.set_epoch(epoch + 1)
            print("Epoch done, recomputing batch sampler")
        start_step = 0


    # df = pd.DataFrame(loss_list)
    # df.to_csv("./loss_curve.csv",index=False)
    

if __name__ == "__main__":
    main()

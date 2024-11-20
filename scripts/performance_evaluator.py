import resource
import inspect
from time import time
from typing import Optional
import multiprocessing as mp

import torch
import torch_musa
from torch.nn import ModuleList
import torch.distributed as dist
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from colossalai.cluster import DistCoordinator

_pipe = None


def divide(x: float, y: float) -> float:
    if y == 0:
        return float("inf")
    elif y == float("inf"):
        return float("nan")
    return x / y


@torch.no_grad()
def all_reduce_mean(x: float, world_size: int, local_only: bool = True) -> float:
    if local_only:
        return x

    if world_size == 1:
        return x
    tensor = torch.tensor([x], device=torch.musa.current_device(), dtype=torch.float)
    dist.all_reduce(tensor)
    tensor = tensor / world_size
    return tensor.item()


def get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = mp.Pipe()
    conn_in, conn_out = _pipe
    return conn_in, conn_out

def run(conn, models, x, t, y, t_tmp, kwarg, recompute):
    """
    Run a single forward test in a subprocess.
    Recomputation is a controllable option.
    Memory results are returned via pipe.
    """
    batch_size, seq_len = x.shape[:2]
    torch.musa.empty_cache()
    torch.musa.reset_peak_memory_stats()
    empty_mem = torch.musa.memory_allocated()
    # assert empty_mem == 0
    if not isinstance(models, ModuleList):
        models = ModuleList([models])
    models = models.musa()
    x = x.musa()
    models.train()
    layer_name = ["norm1", "attn", "attn.qkv", "attn.q_norm", "attn.k_norm", "attn.attn_drop", "attn.proj", "attn.proj_drop", 
                  "cross_attn", "mlp.fc1", "mlp.act", "mlp.drop1", "mlp.norm", "mlp.fc2", "mlp.drop2", 
                  "attn_temp", "attn_temp.qkv", "attn_temp.q_norm", "attn_temp.k_norm", "attn_temp.attn_drop", "attn_temp.proj", "attn_temp.proj_drop"]
    for module in models:
        for name, layer in module.named_modules():
            weight_mem, grad_mem, adam_mem, master_weight_mem = get_mem_stats(layer, torch.bfloat16)
            print(f"{name}: weight_mem {weight_mem/1024**3:.2f} GB; grad_mem {grad_mem/1024**3:.2f} GB; adam_mem {adam_mem/1024**3:.2f} GB;")
        kwargs = {}
        forward_params = inspect.signature(module.forward).parameters
        # if 'attention_mask' in forward_params:
        #     kwargs['attention_mask'] = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        # if 'position_ids' in forward_params:
        #     kwargs['position_ids'] = torch.arange(seq_len, dtype=torch.long,
        #                                           device=x.device).unsqueeze(0).view(-1, seq_len)
        # if 'timestep' in forward_params:
        #     kwargs['timestep'] = torch.randint(0, 1000, (x.shape[0],), device=x.device)
        # if 'y' in forward_params:
        #     kwargs['y'] = torch.rand(batch_size, 1, 120, 4096, device=x.device)
        if 'mask' in forward_params:
            kwargs['mask'] = kwarg['mask']
        if 'x_mask' in forward_params:
            kwargs['x_mask'] = kwarg['x_mask']
        if 't0' in forward_params:
            kwargs['t0'] = kwarg['t0']
        if 't0_tmp' in forward_params:
            kwargs['t0_tmp'] = kwarg['t0_tmp']
        if 'T' in forward_params:
            kwargs['T'] = 16
        if 'S' in forward_params:
            kwargs['S'] = 256
            

        # t_tmp for stdit2
        if recompute:
            # y = checkpoint(module, x, y, t, use_reentrant=False)
            y = checkpoint(module, x, y, t, t_tmp, kwargs['mask'], kwargs['x_mask'], kwargs['t0'], kwargs['t0_tmp'],kwargs['T'], kwargs['S'],  use_reentrant=False)
        else:
            # y = module(x, y, t, **kwargs)
            y = module(x, y, t, t_tmp, kwargs['mask'], kwargs['x_mask'], kwargs['t0'], kwargs['t0_tmp'],kwargs['T'], kwargs['S'])

    torch.musa.synchronize()
    torch.musa.empty_cache()
    peak_mem = torch.musa.max_memory_allocated() - empty_mem
    final_mem = torch.musa.memory_allocated()
    used_mem = final_mem - empty_mem
    # conn.send((used_mem, peak_mem))
    return used_mem, peak_mem

def get_mem_stats(model, dtype):
    """
    Calculate memory statistics for the given module under given precision.
    """
    weight_mem = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_mem = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    adam_mem = sum(p.numel() * 4 for p in model.parameters()) * 2
    master_weight_mem = sum(p.numel() * 4 for p in model.parameters()) if dtype != torch.float32 else 0
    return weight_mem, grad_mem, adam_mem, master_weight_mem


def profile(
    name,
    model,
    x,
    t,
    y,
    t_tmp, 
    kwargs, 
    dtype=torch.float32,
    zero_size=1,
    recompute=False,
    cpu_offload=False,
    verbose=False,
):
    """
    Profile a pytorch module.
    Controllable options:
        dtype: Training precision
        zero_size: the scale of zero data parallelism
        recompute: run forward with no_grad and recompute it right before backward to save activation memory
        cpu_offload: move optimizer memory away from GPU
    Returns:
        model_mem: total memory usage in regard to the parameters of given module, including weights, gradients, Adam momemtums
        act_mem: peak activation memory during training
    """
    if not isinstance(name, (list, tuple)):
        name = [name]
    conn_in, conn_out = get_pipe()
    weight_mem, grad_mem, adam_mem, master_weight_mem = get_mem_stats(model, dtype)
    optimizer_mem = adam_mem + master_weight_mem
    model_mem = weight_mem + grad_mem
    if not cpu_offload:
        model_mem += optimizer_mem / zero_size
    used_mem, peak_mem = run(conn_in, model, x, t, y, t_tmp, kwargs, recompute)
    extra_mem = peak_mem - used_mem
    act_mem = used_mem - weight_mem
    msg = f"[{', '.join(name)}] -- "
    msg += f"weight memory: {weight_mem/1024**3:.2f} GB, "
    msg += "checkpoint" if recompute else "activation"
    msg += f" memory: {act_mem/1024**3:.2f} GB, "
    msg += f"extra memory: {extra_mem/1024**3:.2f} GB\n"
    if verbose:
        print(msg)
    return model_mem, act_mem

class Timer:
    def __init__(self, use_pp=False, grad_accum=False) -> None:
        # self.state: "before_iter" -> "before_forward" -> "before_backward" -> "before_optimizer_update" -> "before_iter"
        # When using pipeline parallel, forward pass and backward pass are entangled, thus skipping calling before_backward()

        self.start_time: Optional[float] = None
        self.last_time_checkpoint: Optional[float] = None
        self.state = "before_iter"
        self.torch_profiler_duration: float = 0.0
        self.data_load_duration: float = 0.0
        self.video_encode_duration: float = 0.0
        self.text_encode_duration: float = 0.0
        self.forward_duration: float = 0.0
        self.backward_duration: float = 0.0
        self.forward_backward_duration: float = 0.0 # Only used when pp is enabled.
        self.optimizer_update_duration: float = 0.0
        self.iter_duration: float = 0.0
        self.use_pp = use_pp
        self.grad_accum = grad_accum
        
    def start(self) -> None:
        assert self.state == "before_iter"
        self.start_time = time()
        self.last_time_checkpoint = self.start_time
        
    def before_video_encode(self, torch_profiler_duration):
        assert self.state == "before_iter"
        self.state = "before_video_encode"
        self.torch_profiler_duration = torch_profiler_duration # The time of torch.profiler.step() shouldn't be considered

        current_time = time()
        self.data_load_duration += current_time - self.last_time_checkpoint - self.torch_profiler_duration
        self.last_time_checkpoint = current_time
    
    def before_text_encode(self,):
        assert self.state == "before_video_encode"
        self.state = "before_text_encode"
        
        current_time = time()
        self.video_encode_duration += current_time - self.last_time_checkpoint
        self.last_time_checkpoint = current_time

    def before_forward(self, torch_profiler_duration: float) -> None:
        # assert self.state == "before_iter"
        # self.state = "before_forward"
        # self.torch_profiler_duration = torch_profiler_duration # The time of torch.profiler.step() shouldn't be considered

        # current_time = time()
        # self.data_load_duration += current_time - self.last_time_checkpoint - self.torch_profiler_duration
        # self.last_time_checkpoint = current_time
        assert self.state == "before_text_encode"
        self.state = "before_forward"
        current_time = time()
        self.text_encode_duration += current_time - self.last_time_checkpoint
        self.last_time_checkpoint = current_time

    def before_backward(self) -> None:
        assert self.state == "before_forward"
        self.state = "before_backward"
        current_time = time()
        self.forward_duration += current_time - self.last_time_checkpoint
        self.last_time_checkpoint = current_time

    def before_optimizer_update(self) -> None:
        if not self.use_pp: # In pipeline parallel, forward and backward are entangled together.
            assert self.state == "before_backward"
            self.state = "before_optimizer_update"
            current_time = time()
            self.backward_duration += current_time - self.last_time_checkpoint
        else:
            assert self.state == "before_forward"
            self.state = "before_optimizer_update"
            current_time = time()
            self.forward_backward_duration += current_time - self.last_time_checkpoint            
        self.last_time_checkpoint = current_time

    def end(self) -> None:
        assert self.start_time is not None

        # When using grad accum, optimizer.step might be skipped.
        # TODO: This assertion should be fixed when implementing benchmarking on pipeline + grad_accum
        assert (self.state ==  "before_optimizer_update") or (self.grad_accum and self.state == "before_backward") 

        current_time = time()
        if self.state == "before_optimizer_update":
            self.optimizer_update_duration += current_time - self.last_time_checkpoint
        elif self.grad_accum and self.state == "before_backward":
            self.backward_duration += current_time - self.last_time_checkpoint
        
        self.state = "before_iter"

        current_iter_duration = current_time - self.start_time - self.torch_profiler_duration
        self.iter_duration += current_iter_duration

        self.start_time = None
        self.last_time_checkpoint = None
        self.torch_profiler_duration = 0.0
        
        return current_iter_duration

    def reset(self) -> None:
        self.data_load_duration = 0.0
        self.forward_duration = 0.0
        self.backward_duration = 0.0
        self.video_encode_duration = 0.0
        self.text_encode_duration = 0.0
        self.forward_backward_duration = 0.0
        self.optimizer_update_duration = 0.0
        self.iter_duration = 0.0
        self.torch_profiler_duration = 0.0
        self.state = "before_iter"

class PerformanceEvaluator:
    """
        Callback for valuate the performance of the model.
    Args:
        actor_num_params: The number of parameters of the actor model.
        critic_num_params: The number of parameters of the critic model.
        initial_model_num_params: The number of parameters of the initial model.
        reward_model_num_params: The number of parameters of the reward model.
        enable_grad_checkpoint: Whether to enable gradient checkpointing.
        ignore_episodes: The number of episodes to ignore when calculating the performance.
    """

    def __init__(
        self,
        model_numel: int,
        stdit_weight_memory: int,
        total_weight_memory: int,
        num_layers: int,
        hidden_size: int,
        vocab_size: int,
        max_seq_length: int,
        num_steps: int,
        use_torch_profiler: bool,
        torch_profiler_path: Optional[str] = None,
        enable_grad_checkpoint: bool = False,
        grad_checkpoint_ratio: float = 1.0,
        ignore_steps: int = 0,
        grad_accum: int = 1,
        include_optimizer_time: bool = False,
        include_data_gen_time: bool = False,
        disable_internal_sync: bool = False,
        dp_size: Optional[int] = None,
        tp_size: int = 1,
        pp_size: int = 1, 
        cfg: dict = None,
        use_t5: bool = True,
    ) -> None:
        self.model_numel = model_numel
        self.max_seq_length = max_seq_length
        self.enable_grad_checkpoint = enable_grad_checkpoint
        self.grad_checkpoint_ratio = grad_checkpoint_ratio
        self.num_steps = num_steps
        self.ignore_steps = ignore_steps
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.coordinator = DistCoordinator()
        self.coordinator.print_on_master(
            "Incorrect grad_checkpoint_ratio might lead to misleading TFLOPS"
        )
        self.grad_accum = grad_accum
        self.include_optimizer_time = include_optimizer_time
        self.include_data_gen_time = include_data_gen_time
        self.disable_internal_sync = disable_internal_sync
        self.dp_size = dp_size or self.coordinator.world_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.mp_world_size = tp_size * pp_size
        self.timer = Timer(use_pp=self.pp_size > 1, grad_accum=(grad_accum > 1))
        self.num_samples: int = 0
        self.flop_megatron = 0
        self.flop: int = 0
        self.flop_hfu: int = 0
        self.t5_flop: float = 17.87 # 16x256x256:bs8: 8.94 T, bs16:17.87 T; 16x512x512: bs2:2.23 TFLOPS; 64x512x512: bs1：5.76 TFLOPS
        self.vae_flop: float = 0.2724  # 16x256x256: bs8: 0.1362 T, bs16: 0.2724 T; 16x512x512:0.3487 TFLOPS; bs2; 64x512x512: bs1:0.6973
        self.stdit_flop: float = 264.03 # 16x256x256: bs8: 132.02 T, bs16: 264.03 T; 16x512x512: bs2:131.67; 64x512x512: bs1: 263.16 TFLOPS
        self.skip_record = True
        self.step_cnt = 0 # The number of benchmarked iterations, should be args.num_steps - args.ignore_steps
        # When opening grad accum, the number of calling optimizer.step might be smaller than self.step_cnt
        self.optimizer_update_cnt = 0
        self.stdit_weight_memory = stdit_weight_memory
        self.total_weight_memory = total_weight_memory
        self.grad_memory = stdit_weight_memory
        self.use_t5 = use_t5
        
        # check input shape
        if cfg.model["type"] == "STDiT-XL/2":
            if cfg.cfg_name == "16x256x256" and cfg.batch_size==8:
                self.t5_flop = 8.94 # 16x256x256:bs8: 8.94 T, 
                self.vae_flop = 34.86  # 16x256x256: bs8: 34.86 T, 
                self.stdit_flop = 44.00 # 16x256x256: bs8: 44.00 T
            elif cfg.cfg_name == "16x256x256" and cfg.batch_size==12:
                self.t5_flop = 13.38
                self.vae_flop = 52.29 
                self.stdit_flop = 66.00 
            elif cfg.cfg_name == "16x256x256" and cfg.batch_size==16:
                self.t5_flop = 17.87  
                self.vae_flop = 69.72  
                self.stdit_flop = 88.25 
            elif cfg.cfg_name == "16x512x512"and cfg.batch_size==2:
                self.t5_flop = 2.23  
                self.vae_flop = 34.87  
                self.stdit_flop = 44.23 
            elif cfg.cfg_name == "16x512x512"and cfg.batch_size==4:
                self.t5_flop = 4.48  
                self.vae_flop = 69.72 
                self.stdit_flop = 44.56 
            elif cfg.cfg_name == "64x512x512":
                self.t5_flop = 1.12 
                self.vae_flop = 69.73  
                self.stdit_flop = 87.72 
        elif cfg.model["type"] == "STDiT2-XL/2":
            if cfg.cfg_name == "16x256x256" and cfg.batch_size==8:
                self.t5_flop = 8.94
                self.vae_flop = 34.86 
                self.stdit_flop = 44.00
                if cfg.hidden_dim == 1536:
                    self.stdit_flop = 20
            elif cfg.cfg_name == "16x256x256" and cfg.batch_size==12:
                self.t5_flop = 13.38
                self.vae_flop = 52.29 
                self.stdit_flop = 66.00 
                if cfg.hidden_dim == 1536:
                    self.stdit_flop = 29.477
            elif cfg.cfg_name == "16x256x256" and cfg.batch_size==16:
                self.t5_flop = 29.89  
                self.vae_flop = 69.73 
                self.stdit_flop = 88.25 
                if cfg.hidden_dim == 1536:
                    self.stdit_flop = 39.9
            elif cfg.cfg_name == "16x512x512"and cfg.batch_size==2:
                self.t5_flop = 2.23  
                self.vae_flop = 34.87 
                self.stdit_flop = 44.23 
                if cfg.hidden_dim == 1536:
                    self.stdit_flop = 19.61
            elif cfg.cfg_name == "16x512x512"and cfg.batch_size==4:
                self.t5_flop = 4.48 
                self.vae_flop = 69.73 
                self.stdit_flop = 44.56
                if cfg.hidden_dim == 1536:
                    self.stdit_flop = 39.22
            elif cfg.cfg_name == "64x512x512":
                self.t5_flop = 1.12 
                self.vae_flop = 69.73   
                self.stdit_flop = 87.72 
                if cfg.hidden_dim == 1536:
                    self.stdit_flop = 78.67
        self.stdit_flop_hfu = self.stdit_flop * 3 + self.stdit_flop * (cfg.num_ckpt_blocks/28) #  1 for fwd + 1 for grad ckpt + 2 for bwd 
        self.stdit_flop = self.stdit_flop * 3 #  1 for fwd + 2 for bwd 
        
        if not self.use_t5:
            self.t5_flop = 0
        
        # Sanity Check
        assert self.dp_size * self.tp_size * self.pp_size == self.coordinator.world_size
        self.torch_profiler = None
        self.torch_profiler_path = torch_profiler_path
        self.use_torch_profiler = use_torch_profiler
        if self.use_torch_profiler:
            assert self.torch_profiler_path is not None

    def on_fit_start(self) -> None:

        # Check Memory Usage before training
        self.optim_init_memory = torch.musa.memory_allocated() - self.total_weight_memory * 1024**3
        # HACK: we assume that the memory occupied by gradients is the same as the memory occupied by weights
        # self.grad_memory = self.total_weight_memory
        self.coordinator.print_on_master(f"Allocated CUDA memory before training: {torch.musa.memory_allocated()/1024**3:.3f} GB")
        self.coordinator.print_on_master(f"Peak CUDA memory before training: {torch.musa.max_memory_allocated()/1024**3:.3f} GB")
        self.coordinator.print_on_master(
            f"Peak CPU memory before training: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.3f} GB"
        )

        # Create a torch profiler
        if self.use_torch_profiler:
            assert self.ignore_steps > 1
            wait_steps = 1
            warmup_steps = self.ignore_steps - wait_steps
            active_steps = self.num_steps - warmup_steps - wait_steps
            self.torch_profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=wait_steps, warmup=warmup_steps, active=active_steps, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.torch_profiler_path),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
            )
            self.torch_profiler.start()

    def step_torch_profiler(self):
        assert self.use_torch_profiler
        self.torch_profiler.step()

    def start_new_iter(self) -> None:
        self.skip_record = (self.ignore_steps > 0 and self.step_cnt < self.ignore_steps)
        if self.skip_record:
            return
        torch.musa.synchronize()
        self.timer.start()
        
    def before_video_encode(self) -> None:
        if self.skip_record:
            return
        if not self.disable_internal_sync:
            torch.musa.synchronize()
        self.timer.before_video_encode(torch_profiler_duration=0)
    
    def before_text_encode(self) -> None:
        if self.skip_record:
            return
        if not self.disable_internal_sync:
            torch.musa.synchronize()
        self.timer.before_text_encode()

    def before_forward(self) -> None:
        if self.skip_record:
            return
        if not self.disable_internal_sync:
            torch.musa.synchronize()
        self.timer.before_forward(torch_profiler_duration=0)

    def before_backward(self) -> None:
        assert self.pp_size == 1, "PerformanceEvaluator.before_backward shouldn't be called when pipeline is enabled"
        if self.skip_record:
            return
        if not self.disable_internal_sync:
            torch.musa.synchronize()
        self.timer.before_backward()

    def before_optimizer_update(self) -> None:
        if self.skip_record:
            return
        if not self.disable_internal_sync:
            torch.musa.synchronize()
        self.timer.before_optimizer_update()
        self.optimizer_update_cnt += 1

    def end_iter(self, input_ids: Tensor, **kwargs) -> None:
        self.step_cnt += 1
        self.coordinator.print_on_master(
            f"\n"
            f"Step: {self.step_cnt - 1}, Is warming up: {self.skip_record}, "
            f"Peak Memory: {torch.musa.max_memory_allocated()/1024**3:.3f} GB, "
            f"Allocated Memory: {torch.musa.memory_allocated()/1024**3:.3f} GB, "
            f"CPU Memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.3f} GB"
        )

        if self.skip_record:
            if self.use_torch_profiler:
                self.step_torch_profiler()
            return

        torch.musa.synchronize()
        current_iter_duration = self.timer.end()

        batch_size, seq_len = input_ids.shape
        self.num_samples += batch_size
        checkpoint_activations_factor = (3 + int(self.enable_grad_checkpoint) * self.grad_checkpoint_ratio)
        self.flop_megatron += (24 * checkpoint_activations_factor * batch_size * seq_len * self.num_layers * (self.hidden_size**2)) * (1. + (seq_len / (6. * self.hidden_size)) + (self.vocab_size / (16. * self.num_layers * self.hidden_size)))
        # flop = batch_size * seq_len * self.model_numel * 2 * checkpoint_activations_factor
        # flop = self.t5_flop + self.vae_flop + self.stdit_flop
        flop = self.stdit_flop # only stidt
        flop_hfu = self.t5_flop + self.vae_flop + self.stdit_flop_hfu
        self.flop += flop
        self.flop_hfu += flop_hfu
        # Reporting speed performance, using statistics on master rank for convenience.
        self.coordinator.print_on_master(
            # f"TGS of last iteration: {batch_size * seq_len / (current_iter_duration + 1e-12) / self.mp_world_size:.3f} tokens/s, "
            f"TGS of last iteration: {batch_size / (current_iter_duration + 1e-12):.3f} samples/s, "
            f"TFLOPS of last iteration: {flop / (current_iter_duration + 1e-12):.3f}"
        )

        if self.use_torch_profiler:
            self.step_torch_profiler()

    def on_fit_end(self) -> None:

        # End torch profiler
        if self.use_torch_profiler:
            self.torch_profiler.stop()

        with open(f"memory_{dist.get_rank()}.log", "w") as f:
            f.write(torch.musa.memory_summary(device=torch.musa.current_device()))

        if dist.get_rank() != 0:
            return

        # Overall Stats
        num_record_steps = self.step_cnt - self.ignore_steps if (self.step_cnt - self.ignore_steps ) > 0 else 1
        iter_duration = self.timer.iter_duration
        if not self.include_optimizer_time:
            iter_duration -= self.timer.optimizer_update_duration
        if not self.include_data_gen_time:
            iter_duration -= self.timer.data_load_duration
        
        # rm random data generate in train with random dataset
        avg_duration = all_reduce_mean(iter_duration, self.coordinator.world_size)
        avg_latency = avg_duration / num_record_steps
        avg_throughput = self.num_samples * self.dp_size / (avg_duration + 1e-12)
        # tokens_per_gpu_per_second = self.num_samples * self.max_seq_length / (avg_duration + 1e-12) / self.mp_world_size
        # avg_tflops_per_gpu_megatron = self.flop_megatron / 1e12 / (avg_duration + 1e-12) / self.mp_world_size
        avg_stdit_tflops_per_gpu = self.stdit_flop / (self.timer.forward_duration + self.timer.backward_duration + 1e-12) 
        avg_t5_tflops_per_gpu = self.t5_flop  / (self.timer.text_encode_duration + 1e-12) 
        avg_vae_tflops_per_gpu = self.vae_flop  / (self.timer.video_encode_duration + 1e-12) 
        avg_tflops_per_gpu = self.flop /num_record_steps/ (avg_latency + 1e-12)
        avg_tflops_hfu_per_gpu = self.flop_hfu /num_record_steps/ (avg_latency + 1e-12)
        self.coordinator.print_on_master(
            f"Overall Stats: "
            f"batch_size_per_device: {self.num_samples / num_record_steps}, sequence_length: {self.max_seq_length}, dp_size: {self.dp_size}, "
            f"Latency: {avg_latency:.3f} s, Throughput: {avg_throughput:.3f} samples/sec, "
            # f"Latency: {avg_latency:.3f} s, Throughput: {avg_throughput:.3f} samples/sec, TGS: {tokens_per_gpu_per_second:.3f} tokens/s, "
            f"STDIT TFLOPS: {avg_stdit_tflops_per_gpu:.3f}, T5 encoder TFLOPS: {avg_t5_tflops_per_gpu:.3f}, Vae TFLOPS: {avg_vae_tflops_per_gpu:.3f}, "
            f"TFLOPS per GPU (MFU): {avg_tflops_per_gpu:.3f}, TFLOPS per GPU (HFU): {avg_tflops_hfu_per_gpu:.3f},"
            f"\n"
        )

        # Time Breakdown Stats
        data_load_duration = all_reduce_mean(self.timer.data_load_duration, self.coordinator.world_size)
        if self.pp_size == 1:
            video_encode_duration = all_reduce_mean(self.timer.video_encode_duration, self.coordinator.world_size)
            text_encode_duration = all_reduce_mean(self.timer.text_encode_duration, self.coordinator.world_size)
            forward_duration = all_reduce_mean(self.timer.forward_duration, self.coordinator.world_size)
            backward_duration = all_reduce_mean(self.timer.backward_duration, self.coordinator.world_size)
        else:
            forward_backward_duration = all_reduce_mean(self.timer.forward_backward_duration, self.coordinator.world_size)
        optimizer_update_duration = all_reduce_mean(self.timer.optimizer_update_duration, self.coordinator.world_size)

        time_usage_log = f"Time Usage Breakdown: "
        time_usage_log += f"Avg Dataload Latency: {1000 * data_load_duration / num_record_steps:.2f} ms, "
        if self.pp_size == 1:
            time_usage_log += f"Avg Video Encode Latency: {1000 * video_encode_duration / num_record_steps:.2f} ms, "
            time_usage_log += f"Avg Text Encode Latency: {1000 * text_encode_duration / num_record_steps:.2f} ms, "
            time_usage_log += f"Avg Forward Latency: {1000 * forward_duration / num_record_steps:.2f} ms, "
            time_usage_log += f"Avg Backward Latency: {1000 * backward_duration / num_record_steps:.2f} ms, "
        else:
            time_usage_log += f"Avg Forward Backward Latency: {1000 * forward_backward_duration / num_record_steps:.2f} ms, "
        if self.optimizer_update_cnt > 0:
            time_usage_log += f"Avg Optimizer Update Latency: {1000 * optimizer_update_duration / self.optimizer_update_cnt:.2f} ms, "
        time_usage_log += f"Avg Step Latency: {1000 * avg_latency:.2f} ms\n"
        self.coordinator.print_on_master(time_usage_log)
        
        peak_memory = torch.musa.max_memory_allocated()
        torch.musa.empty_cache()
        memory_fragmentation = torch.musa.max_memory_reserved() - peak_memory
        final_allocated_memory = torch.musa.memory_allocated()
        optimizer_memory = final_allocated_memory - self.total_weight_memory * 1024**3
        assert optimizer_memory >= self.optim_init_memory, "Optimizer memory should be larger than the initial memory"
        activation_memory = peak_memory - final_allocated_memory - self.grad_memory
        self.coordinator.print_on_master(
            f"Memory Usage Breakdown: "
            f"Stdit Weight {self.stdit_weight_memory :.3f} GB,"
            f"Total Weight {self.total_weight_memory :.3f} GB, "
            f"Grad {self.grad_memory:.3f} GB, "
            f"Optimizer {optimizer_memory/1024**3:.3f} GB, "
            f"Activation {activation_memory/1024**3:.3f} GB, "
            f"Peak {peak_memory/1024**3:.3f} GB, "
            f"Frag {memory_fragmentation/1024**3:.3f} GB, "
            f"Final {final_allocated_memory/1024**3:.3f} GB, "
            f"CPU {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.3f} GB"
            F"\n"
        )
        # self.coordinator.print_on_master(
        #     "Notice: Sometimes the weight and optimizer are initialized together (e.g. booster.boost/deepspeed.initialize), "
        #     "in such cases the calculated Weight memory is the sum of model weight memory and part of optimizer memory, please refer to other logs for model weight memory information."
        # )
        self.coordinator.print_on_master(torch.musa.memory_summary(device=torch.musa.current_device()))


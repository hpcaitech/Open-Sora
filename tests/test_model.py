import colossalai
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from colossalai.booster import Booster
from colossalai.logging import disable_existing_loggers
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device
from torch.testing import assert_close

from open_sora.modeling import DiT_models
from open_sora.utils.plugin import ZeroSeqParallelPlugin


@parameterize("sp_size", [2, 4])
def check_dit_model_fwd_bwd(
    sp_size: int, video_latent_states, text_latent_states, t, mask
):
    plugin = ZeroSeqParallelPlugin(
        sp_size=sp_size, stage=2, precision="fp32", master_weights=False
    )
    booster = Booster(plugin=plugin)
    model = DiT_models["DiT-B/8"](text_dropout_prob=0.0).to(get_current_device())
    parallel_model = DiT_models["DiT-B/8"](
        text_dropout_prob=0.0, seq_parallel_group=plugin.sp_group
    ).to(get_current_device())
    parallel_model.load_state_dict(model.state_dict())
    opt = HybridAdam(parallel_model.parameters(), lr=1e-3)
    parallel_model, opt, *_ = booster.boost(parallel_model, opt)

    target = model(video_latent_states, t, text_latent_states, mask)
    noise = torch.randn_like(target)
    target_loss = F.mse_loss(target, noise)
    target_loss.backward()

    dp_video_latent_states = video_latent_states.chunk(plugin.dp_size)[plugin.dp_rank]
    dp_text_latent_states = text_latent_states.chunk(plugin.dp_size)[plugin.dp_rank]
    dp_t = t.chunk(plugin.dp_size)[plugin.dp_rank]
    dp_mask = mask.chunk(plugin.dp_size)[plugin.dp_rank]
    dp_noise = noise.chunk(plugin.dp_size)[plugin.dp_rank]

    output = parallel_model(
        dp_video_latent_states, dp_t, dp_text_latent_states, dp_mask
    )
    loss = F.mse_loss(output, dp_noise)
    booster.backward(loss, opt)

    if plugin.dp_size == 1:
        assert_close(target, output)

    for p1, p2 in zip(model.parameters(), opt._master_param_groups_of_current_rank[0]):
        working_p = opt._param_store.master_to_working_param[id(p2)]
        grads = opt._grad_store.get_partitioned_gradients_by_param_id(0, id(working_p))
        grad_index = 0 if opt._partition_grads else opt._local_rank
        grad = grads[grad_index]
        sharded_grad = p1.grad.view(-1).chunk(dist.get_world_size())[dist.get_rank()]
        assert_close(sharded_grad, grad[: sharded_grad.shape[0]])


def run_dist(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(
        config={},
        rank=rank,
        world_size=world_size,
        port=port,
        host="localhost",
        backend="nccl",
    )
    b, s, c, p = 4, 20, 3, 8
    dim_text, s_text = 512, 12
    video_latent_states = torch.rand(b, s, c, p, p, device=get_current_device())
    text_latent_states = torch.rand(b, s_text, dim_text, device=get_current_device())
    t = torch.randint(0, 1000, (b,), device=get_current_device())
    mask = torch.ones(b, 1, s, s_text, device=get_current_device(), dtype=torch.int)
    check_dit_model_fwd_bwd(
        video_latent_states=video_latent_states,
        text_latent_states=text_latent_states,
        t=t,
        mask=mask,
    )


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_dit_model():
    spawn(run_dist, 4)


if __name__ == "__main__":
    test_dit_model()

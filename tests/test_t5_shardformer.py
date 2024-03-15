import time
from copy import deepcopy

import colossalai
import torch
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import spawn

from opensora.acceleration.shardformer.policy.t5_encoder import T5EncoderPolicy
from opensora.models.text_encoder.t5 import T5Embedder


def run_t5_encoder(rank, world_size, port):
    colossalai.launch({}, rank=rank, world_size=world_size, port=port, host="localhost")

    # t5 embedder
    t5_path = "./pretrained_models/t5_ckpts"
    hf_t5 = T5Embedder(device="cuda", local_cache=True, cache_dir=t5_path, torch_dtype=torch.float)
    sf_t5 = deepcopy(hf_t5)

    # create huggingface model as normal
    shard_config = ShardConfig(
        tensor_parallel_process_group=None,
        pipeline_stage_manager=None,
        enable_tensor_parallelism=False,
        enable_fused_normalization=False,
        enable_flash_attention=False,
        enable_jit_fused=True,
        enable_sequence_parallelism=False,
        enable_sequence_overlap=False,
    )
    shard_former = ShardFormer(shard_config=shard_config)
    sharded_model, _ = shard_former.optimize(sf_t5.model, policy=T5EncoderPolicy())
    sf_t5.model = sharded_model

    # test t5 embedder
    texts = ["Who is the best player in the history of NBA?", "How to study computer science?"]
    for i in range(5):
        hf_embs, hf_masks = hf_t5.get_text_embeddings(texts)
        sf_embs, sf_masks = sf_t5.get_text_embeddings(texts)

    # check accuracy
    assert torch.allclose(hf_embs, sf_embs, rtol=1e-4, atol=1e-5), f"{hf_embs} \nvs\n{sf_embs}"
    assert torch.allclose(hf_masks, sf_masks), f"{hf_masks} \nvs\n{sf_masks}"

    # measure perf
    torch.cuda.synchronize()
    hf_start = time.time()
    for i in range(20):
        hf_embs, hf_masks = hf_t5.get_text_embeddings(texts)
    torch.cuda.synchronize()
    hf_end = time.time()

    # convert sf to fp16
    hf_t5.model = hf_t5.model.half()
    torch.cuda.synchronize()
    sf_start = time.time()
    for i in range(20):
        hf_embs, hf_masks = hf_t5.get_text_embeddings(texts)
    torch.cuda.synchronize()
    sf_end = time.time()

    print(f"[Performance] native: {hf_end - hf_start}s, shardformer: {sf_end - sf_start} s")


def test_t5_encoder():
    spawn(run_t5_encoder)


if __name__ == "__main__":
    test_t5_encoder()

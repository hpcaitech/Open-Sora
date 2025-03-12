plugin = "hybrid"
plugin_config = dict(
    tp_size=1,
    pp_size=1,
    sp_size=8,
    sequence_parallelism_mode="ring_attn",
    enable_sequence_parallelism=True,
    static_graph=True,
    zero_stage=2,
    overlap_allgather=False,
)

plugin_ae = "hybrid"
plugin_config_ae = dict(
    tp_size=8,
    pp_size=1,
    sp_size=1,
    zero_stage=2,
    overlap_allgather=False,
)

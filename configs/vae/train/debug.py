_base_ = ["hunyuan_video_disc.py"]

# model = dict(
#     # use_temporal_tiling = True,
#     use_slicing = True,
#     use_spatial_tiling = True,
#     sample_tsize = 32,
# )

grad_checkpoint = True
# grad_checkpoint_buffer_size = 15 * 1024**3

bucket_config = {
    "_delete_": True,
    "512px_ar1:1": {129: (1.0, 1)},
    # "360p": {33: (1.0, 1)}
}

mixed_image_ratio = 0.0

log_every = 1

plugin = "hybrid"
plugin_config = dict(
    tp_size=8,
    pp_size=1,
    zero_stage=2,
    static_graph=True,
    # sp_size=1,
    # sequence_parallelism_mode="ring_attn",
    # enable_sequence_parallelism=True,
)
ema_decay = None

disc_plugin = "zero2"
disc_plugin_config = dict(
    reduce_bucket_size_in_m=128,
    overlap_allgather=False,
)

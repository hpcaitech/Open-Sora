_base_ = ["stage1.py"]

dataset = dict(memory_efficient=False)

# Define model components
model = dict(
    cond_embed=True,
)

condition_config = dict(
    t2v=1,
    i2v_head=5,
    i2v_loop=1,
    i2v_tail=1,
)
is_causal_vae = True

lr = 1e-5
optim = dict(
    lr=lr,
)
ema_decay = None
async_io = False

plugin = "hybrid"
plugin_config = dict(
    tp_size=1,
    pp_size=1,
    sp_size=1,
    zero_stage=2,
)

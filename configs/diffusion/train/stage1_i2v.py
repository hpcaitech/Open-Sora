_base_ = ["stage1.py"]

# Define model components
model = dict(cond_embed=True)

condition_config = dict(
    t2v=1,
    i2v_head=5,  # train i2v (image as first frame) with weight 5
    i2v_loop=1,  # train image connection with weight 1
    i2v_tail=1,  # train i2v (image as last frame) with weight 1
)

lr = 1e-5
optim = dict(lr=lr)

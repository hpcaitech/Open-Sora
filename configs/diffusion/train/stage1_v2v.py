_base_ = ["stage1_i2v.py"]

condition_config = dict(
    t2v=1,
    i2v_head=5,
    i2v_loop=1,
    i2v_tail=1,
    v2v_head=1,
    v2v_head_easy=1,
    v2v_tail=0.5,
    v2v_tail_easy=0.5,
)

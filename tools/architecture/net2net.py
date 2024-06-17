"""
Implementation of Net2Net (http://arxiv.org/abs/1511.05641)
Numpy modules for Net2Net
- Net2Wider
- Net2Deeper

Written by Kyunghyun Paeng

"""


def net2net(teach_param, stu_param):
    # teach param with shape (a, b)
    # stu param with shape (c, d)
    # net to net (a, b) -> (c, d) where c >= a and d >= b
    teach_param_shape = teach_param.shape
    stu_param_shape = stu_param.shape

    if len(stu_param_shape) > 2:
        teach_param = teach_param.reshape(teach_param_shape[0], -1)
        stu_param = stu_param.reshape(stu_param_shape[0], -1)

    assert len(stu_param.shape) == 1 or len(stu_param.shape) == 2, "teach_param and stu_param must be 2-dim array"
    assert len(teach_param_shape) == len(stu_param_shape), "teach_param and stu_param must have same dimension"

    if len(teach_param_shape) == 1:
        stu_param[: teach_param_shape[0]] = teach_param
    elif len(teach_param_shape) == 2:
        stu_param[: teach_param_shape[0], : teach_param_shape[1]] = teach_param
    else:
        breakpoint()

    if stu_param.shape != stu_param_shape:
        stu_param = stu_param.reshape(stu_param_shape)

    return stu_param


if __name__ == "__main__":
    """Net2Net Class Test"""

    import torch

    from opensora.models.pixart import PixArt_1B_2

    model = PixArt_1B_2(no_temporal_pos_emb=True, space_scale=4, enable_flash_attn=True, enable_layernorm_kernel=True)
    print("load model done")

    ckpt = torch.load("/home/zhouyukun/projs/opensora/pretrained_models/PixArt-Sigma-XL-2-2K-MS.pth")
    print("load ckpt done")

    ckpt = ckpt["state_dict"]
    ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)

    missing_keys = []
    for name, module in model.named_parameters():
        if name in ckpt:
            teach_param = ckpt[name].data
            stu_param = module.data
            stu_param = net2net(teach_param, stu_param)

            module.data = stu_param

            print("processing layer: ", name, "shape: ", module.size())

        else:
            # print("Missing key: ", name)
            missing_keys.append(name)

    print(missing_keys)

    breakpoint()
    torch.save({"state_dict": model.state_dict()}, "PixArt-1B-2.pth")

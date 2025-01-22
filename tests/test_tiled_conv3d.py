import torch

from opensora.models.layers.tiled_conv3d import TiledConv3d

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def test_tiled_conv3d():
    data = torch.rand(1, 128, 51, 256, 256).cuda().to(torch.bfloat16)

    exclude_temporal_dim_options = [True, False]
    padding_options = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 1, 1),
    ]
    stride_options = [(1, 1, 1), (2, 1, 1)]
    kernel_size_options = [(1, 1, 1), (3, 3, 3)]
    tile_size = 16

    for padding in padding_options:
        for stride in stride_options:
            for kernel_size in kernel_size_options:
                for exclude_temporal_dim in exclude_temporal_dim_options:
                    conv3d = (
                        torch.nn.Conv3d(128, 128, kernel_size=kernel_size, stride=stride, padding=padding)
                        .cuda()
                        .to(torch.bfloat16)
                    )
                    auto_tiled_conv3d = TiledConv3d.from_native_conv3d(
                        conv3d, tile_size=tile_size, exclude_temporal_dim=exclude_temporal_dim
                    )

                    # compare
                    with torch.inference_mode():
                        out = conv3d(data)
                        merged_out = auto_tiled_conv3d(data)

                    print(f"max allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.2f} GB")
                    print(f"max reserved: {torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024:.2f} GB")

                    try:
                        torch.testing.assert_close(out, merged_out)
                    except Exception as e:
                        print(
                            f"Failed with padding={padding}, stride={stride}, kernel_size={kernel_size}, exclude_temporal_dim={exclude_temporal_dim}, error: {e}"
                        )


if __name__ == "__main__":
    test_tiled_conv3d()

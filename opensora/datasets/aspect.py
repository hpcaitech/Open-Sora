import math
import os

ASPECT_RATIO_LD_LIST = [  # width:height
    "2.39:1",  # cinemascope, 2.39
    "2:1",  # rare, 2
    "16:9",  # rare, 1.89
    "1.85:1",  # american widescreen, 1.85
    "9:16",  # popular, 1.78
    "5:8",  # rare, 1.6
    "3:2",  # rare, 1.5
    "4:3",  # classic, 1.33
    "1:1",  # square
]


def get_ratio(name: str) -> float:
    width, height = map(float, name.split(":"))
    return height / width


def get_aspect_ratios_dict(
    total_pixels: int = 256 * 256, training: bool = True
) -> dict[str, tuple[int, int]]:
    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))
    aspect_ratios_dict = {}
    aspect_ratios_vertical_dict = {}
    for ratio in ASPECT_RATIO_LD_LIST:
        width_ratio, height_ratio = map(float, ratio.split(":"))
        width = int(math.sqrt(total_pixels * (width_ratio / height_ratio)) // D) * D
        height = int((total_pixels / width) // D) * D

        if training:
            # adjust aspect ratio to match total pixels
            diff = abs(height * width - total_pixels)
            candidate = [
                (height - D, width),
                (height + D, width),
                (height, width - D),
                (height, width + D),
            ]
            for h, w in candidate:
                if abs(h * w - total_pixels) < diff:
                    height, width = h, w
                    diff = abs(h * w - total_pixels)

        # remove duplicated aspect ratio
        if (height, width) not in aspect_ratios_dict.values() or not training:
            aspect_ratios_dict[ratio] = (height, width)
            vertial_ratios = ":".join(ratio.split(":")[::-1])
            aspect_ratios_vertical_dict[vertial_ratios] = (width, height)

    aspect_ratios_dict.update(aspect_ratios_vertical_dict)

    return aspect_ratios_dict


def get_num_pexels(aspect_ratios_dict: dict[str, tuple[int, int]]) -> dict[str, int]:
    return {ratio: h * w for ratio, (h, w) in aspect_ratios_dict.items()}


def get_num_tokens(aspect_ratios_dict: dict[str, tuple[int, int]]) -> dict[str, int]:
    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))
    return {ratio: h * w // D // D for ratio, (h, w) in aspect_ratios_dict.items()}


def get_num_pexels_from_name(resolution: str) -> int:
    resolution = resolution.split("_")[0]
    if resolution.endswith("px"):
        size = int(resolution[:-2])
        num_pexels = size * size
    elif resolution.endswith("p"):
        size = int(resolution[:-1])
        num_pexels = int(size * size / 9 * 16)
    else:
        raise ValueError(f"Invalid resolution {resolution}")
    return num_pexels


def get_resolution_with_aspect_ratio(
    resolution: str,
) -> tuple[int, dict[str, tuple[int, int]]]:
    """Get resolution with aspect ratio

    Args:
        resolution (str): resolution name. The format is name only or "{name}_{setting}".
            name supports "256px" or "360p". setting supports "ar1:1" or "max".

    Returns:
        tuple[int, dict[str, tuple[int, int]]]: resolution with aspect ratio
    """
    keys = resolution.split("_")
    if len(keys) == 1:
        resolution = keys[0]
        setting = ""
    else:
        resolution, setting = keys
        assert setting == "max" or setting.startswith(
            "ar"
        ), f"Invalid setting {setting}"

    # get resolution
    num_pexels = get_num_pexels_from_name(resolution)

    # get aspect ratio
    aspect_ratio_dict = get_aspect_ratios_dict(num_pexels)

    # handle setting
    if setting == "max":
        aspect_ratio = max(
            aspect_ratio_dict,
            key=lambda x: aspect_ratio_dict[x][0] * aspect_ratio_dict[x][1],
        )
        aspect_ratio_dict = {aspect_ratio: aspect_ratio_dict[aspect_ratio]}
    elif setting.startswith("ar"):
        aspect_ratio = setting[2:]
        assert (
            aspect_ratio in aspect_ratio_dict
        ), f"Aspect ratio {aspect_ratio} not found"
        aspect_ratio_dict = {aspect_ratio: aspect_ratio_dict[aspect_ratio]}

    return num_pexels, aspect_ratio_dict


def get_closest_ratio(height: float, width: float, ratios: dict) -> str:
    aspect_ratio = height / width
    closest_ratio = min(
        ratios.keys(), key=lambda ratio: abs(aspect_ratio - get_ratio(ratio))
    )
    return closest_ratio


def get_image_size(
    resolution: str, ar_ratio: str, training: bool = True
) -> tuple[int, int]:
    num_pexels = get_num_pexels_from_name(resolution)
    ar_dict = get_aspect_ratios_dict(num_pexels, training)
    assert ar_ratio in ar_dict, f"Aspect ratio {ar_ratio} not found"
    return ar_dict[ar_ratio]


def bucket_to_shapes(bucket_config, batch_size=None):
    shapes = []
    for resolution, infos in bucket_config.items():
        for num_frames, (_, bs) in infos.items():
            aspect_ratios = get_aspect_ratios_dict(get_num_pexels_from_name(resolution))
            for ar, (height, width) in aspect_ratios.items():
                if batch_size is not None:
                    bs = batch_size
                shapes.append((bs, 3, num_frames, height, width))
    return shapes

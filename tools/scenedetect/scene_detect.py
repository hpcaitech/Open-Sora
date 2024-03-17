import os
from multiprocessing import Pool

from mmengine.logging import MMLogger
from scenedetect import ContentDetector, detect
from tqdm import tqdm

from opensora.utils.misc import get_timestamp

from .utils import check_mp4_integrity, clone_folder_structure, iterate_files, split_video

# config
target_fps = 30  # int
shorter_size = 512  # int
min_seconds = 1  # float
max_seconds = 5  # float
assert max_seconds > min_seconds
cfg = dict(
    target_fps=target_fps,
    min_seconds=min_seconds,
    max_seconds=max_seconds,
    shorter_size=shorter_size,
)


def process_folder(root_src, root_dst):
    # create logger
    folder_path_log = os.path.dirname(root_dst)
    log_name = os.path.basename(root_dst)
    timestamp = get_timestamp()
    log_path = os.path.join(folder_path_log, f"{log_name}_{timestamp}.log")
    logger = MMLogger.get_instance(log_name, log_file=log_path)

    # clone folder structure
    clone_folder_structure(root_src, root_dst)

    # all source videos
    mp4_list = [x for x in iterate_files(root_src) if x.endswith(".mp4")]
    mp4_list = sorted(mp4_list)

    for idx, sample_path in tqdm(enumerate(mp4_list)):
        folder_src = os.path.dirname(sample_path)
        folder_dst = os.path.join(root_dst, os.path.relpath(folder_src, root_src))

        # check src video integrity
        if not check_mp4_integrity(sample_path, logger=logger):
            continue

        # detect scenes
        scene_list = detect(sample_path, ContentDetector(), start_in_scene=True)

        # split scenes
        save_path_list = split_video(sample_path, scene_list, save_dir=folder_dst, **cfg, logger=logger)

        # check integrity of generated clips
        for x in save_path_list:
            check_mp4_integrity(x, logger=logger)


def scene_detect():
    """detect & cut scenes using a single process
    Expected dataset structure:
    data/
        your_dataset/
            raw_videos/
                xxx.mp4
                yyy.mp4

    This function results in:
    data/
        your_dataset/
            raw_videos/
                xxx.mp4
                yyy.mp4
                zzz.mp4
            clips/
                xxx_scene-0.mp4
                yyy_scene-0.mp4
                yyy_scene-1.mp4
    """
    # TODO: specify your dataset root
    root_src = f"./data/your_dataset/raw_videos"
    root_dst = f"./data/your_dataset/clips"

    process_folder(root_src, root_dst)


def scene_detect_mp():
    """detect & cut scenes using multiple processes
    Expected dataset structure:
    data/
        your_dataset/
            raw_videos/
                split_0/
                    xxx.mp4
                    yyy.mp4
                split_1/
                    xxx.mp4
                    yyy.mp4

    This function results in:
    data/
        your_dataset/
            raw_videos/
                split_0/
                    xxx.mp4
                    yyy.mp4
                split_1/
                    xxx.mp4
                    yyy.mp4
            clips/
                split_0/
                    xxx_scene-0.mp4
                    yyy_scene-0.mp4
                split_1/
                    xxx_scene-0.mp4
                    yyy_scene-0.mp4
                    yyy_scene-1.mp4
    """
    # TODO: specify your dataset root
    root_src = f"./data/your_dataset/raw_videos"
    root_dst = f"./data/your_dataset/clips"

    # TODO: specify your splits
    splits = ["split_0", "split_1"]

    # process folders
    root_src_list = [os.path.join(root_src, x) for x in splits]
    root_dst_list = [os.path.join(root_dst, x) for x in splits]

    with Pool(processes=len(splits)) as pool:
        pool.starmap(process_folder, list(zip(root_src_list, root_dst_list)))


if __name__ == "__main__":
    # TODO: choose single process or multiprocessing
    scene_detect()
    # scene_detect_mp()

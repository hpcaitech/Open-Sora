import argparse
import json
import os
from typing import Dict, Tuple
import warnings
from tqdm import tqdm
import shutil
import multiprocessing


DEFAULT_TYPES = ["train", "val", "test"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-path", type=str, help="The path to the MSR-VTT dataset")
    parser.add_argument("-o", "--output-path", type=str, help="The output to the collated MSR-VTT dataset")
    return parser.parse_args()


def get_annotations(root_path: str):
    """
    Get the annotation data from the MSR-VTT dataset. The annotations are in the format of:

    {
        "annotations": [
            {
                "image_id": "video1",
                "caption": "some
            }
        ]
    }

    Args:
        root_path (str): The root path to the MSR-VTT dataset
    """
    annotation_json_file = os.path.join(root_path, "annotation/MSR_VTT.json")
    with open(annotation_json_file, 'r') as f:
        data = json.load(f)
    return data

def get_video_list(root_path: str, dataset_type: str):
    """
    Get the list of videos in the dataset split.

    Args:
        root_path (str): The root path to the MSR-VTT dataset
        dataset_type (str): The dataset split type. It should be one of "train", "val", or "test"
    """
    assert dataset_type in DEFAULT_TYPES, f"Expected the dataset type to be in {DEFAULT_TYPES}, but got {dataset_type}"
    dataset_file_path = os.path.join(root_path, f"structured-symlinks/{dataset_type}_list_full.txt")
    with open(dataset_file_path, 'r') as f:
        video_list = f.readlines()
        video_list = [x.strip() for x in video_list]
    return video_list


def copy_video(video_id: str, root_path: str, output_path: str, dataset_type: str):
    """
    Copy the video from the source path to the destination path.

    Args:
        video_id (str): The video id
        root_path (str): The root path to the MSR-VTT dataset
        output_path (str): The output path to the collated MSR-VTT dataset
        dataset_type (str): The dataset split type. It should be one of "train", "val", or "test"
    """
    assert dataset_type in DEFAULT_TYPES, f"Expected the dataset type to be in {DEFAULT_TYPES}, but got {dataset_type}"
    src_file = os.path.join(root_path, f"videos/all/{video_id}.mp4")
    dst_folder = os.path.join(output_path, f"{dataset_type}/videos")
    dst_file = os.path.join(dst_folder, f'{video_id}.mp4')
    os.makedirs(dst_folder, exist_ok=True)

    # create symlink
    assert os.path.isfile(src_file), f"Expected the source file {src_file} to exist"
    if not os.path.islink(dst_file):
        shutil.copy(src_file, dst_file)

def get_annotation_file_path(output_path: str, dataset_type: str):
    file_path = os.path.join(output_path, f"{dataset_type}/annotations.json")
    return file_path

def collate_annotation_files(
        annotations: Dict, 
        root_path: str, 
        output_path: str,
        ):
    """
    Collate the video and caption data into a single folder.

    Args:
        annotations (Dict): The annotations data
        root_path (str): The root path to the MSR-VTT dataset
        output_path (str): The output path to the collated MSR-VTT dataset
    """
    # get all video list
    train_video_list = get_video_list(root_path, "train")
    val_video_list = get_video_list(root_path, "val")
    test_video_list = get_video_list(root_path, "test")

    # iterate over annotations
    collated_train_data = []
    collated_val_data = []
    collated_test_data = []

    print("Collating annotations files")

    for anno in tqdm(annotations['annotations']):
        video_id = anno['image_id']
        caption = anno['caption']

        obj = {
            "file": f"{video_id}.mp4",
            "captions": [caption]
        }

        if video_id in train_video_list:
            collated_train_data.append(obj)
        elif video_id in val_video_list:
            collated_val_data.append(obj)
        elif video_id in test_video_list:
            collated_test_data.append(obj)
        else:
            warnings.warn(f"Video {video_id} not found in any of the dataset splits")
    
    def _save_caption_files(obj, dataset_type):
        dst_file = get_annotation_file_path(output_path, dataset_type)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        with open(dst_file, 'w') as f:
            json.dump(obj, f, indent=4)
        
    _save_caption_files(collated_train_data, "train")
    _save_caption_files(collated_val_data, "val")
    _save_caption_files(collated_test_data, "test")

def copy_file(path_pair: Tuple[str, str]):
    src_path, dst_path = path_pair
    shutil.copyfile(src_path, dst_path)

def copy_videos(root_path: str, output_path: str, num_workers: int = 8):
    """
    Batch copy the video files to the output path.

    Args:
        root_path (str): The root path to the MSR-VTT dataset
        output_path (str): The output path to the collated MSR-VTT dataset
        num_workers (int): The number of workers to use for the copy operation
    """
    pool = multiprocessing.Pool(num_workers)

    for dataset_type in DEFAULT_TYPES:
        print(f"Copying videos for the {dataset_type} dataset")
        annotation_file_path = get_annotation_file_path(output_path, dataset_type)
        output_video_folder_path = os.path.join(output_path, f"{dataset_type}/videos")
        os.makedirs(output_video_folder_path, exist_ok=True)

        with open(annotation_file_path, 'r') as f:
            annotation_data = json.load(f)
        
        video_ids = [obj['file'] for obj in annotation_data]
        unique_video_ids = list(set(video_ids))

        path_pairs = [
            (
                os.path.join(root_path, f"videos/all/{video_id}"),
                os.path.join(output_video_folder_path, video_id)
            )
            for video_id in unique_video_ids
        ]

        for _ in tqdm(pool.imap_unordered(copy_file, path_pairs), total=len(path_pairs)):
            pass

def main():
    args = parse_args()
    annotations = get_annotations(args.data_path)
    collate_annotation_files(annotations, args.data_path, args.output_path)
    copy_videos(args.data_path, args.output_path)


if __name__ == "__main__":
    main()
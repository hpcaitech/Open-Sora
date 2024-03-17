import argparse
import csv
import os

from tqdm import tqdm

# path, name, #frames
PREFIX = [
    "The video shows",
    "The video captures",
    "The video features",
    "The video depicts",
    "The video presents",
    "The video features",
    "The video is ",
    "In the video,",
]


def get_video_length(path):
    import cv2

    cap = cv2.VideoCapture(path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def main(args):
    input_path = args.input
    output_path = args.output
    if output_path is None:
        name = os.path.basename(input_path)
        name, ext = os.path.splitext(name)
        if args.num_frames_min is not None:
            name += f"_fmin_{args.num_frames_min}"
        if args.num_frames_max is not None:
            name += f"_fmax_{args.num_frames_max}"
        if args.filter_null_text:
            name += "_fnt"
        if args.remove_prefix:
            name += "_rp"
        if args.root is not None:
            name += f"_root"
        if args.relength:
            name += "_relength"
        output_path = os.path.join(os.path.dirname(input_path), name + ext)

    with open(input_path, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    print("Number of videos before filtering:", len(data))

    data_new = []
    for i, row in tqdm(enumerate(data)):
        path = row[0]
        caption = row[1]
        n_frames = int(row[2])
        if args.num_frames_min is not None and n_frames < args.num_frames_min:
            continue
        if args.num_frames_max is not None and n_frames > args.num_frames_max:
            continue
        if args.filter_null_text and len(caption) == 0:
            continue
        if args.remove_prefix:
            for prefix in PREFIX:
                if caption.startswith(prefix):
                    caption = caption[len(prefix) :].strip()
                    if caption[0].islower():
                        caption = caption[0].upper() + caption[1:]
                    row[1] = caption
                    break
        if args.root is not None:
            row[0] = os.path.join(args.root, path)
        if args.relength:
            n_frames = get_video_length(row[0])
            row[2] = n_frames
        data_new.append(row)

    print("Number of videos after filtering:", len(data_new))
    with open(output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data_new)
    print("Output saved to", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num_frames_min", type=int, default=None)
    parser.add_argument("--num_frames_max", type=int, default=None)
    parser.add_argument("--filter_null_text", action="store_true")
    parser.add_argument("--remove_prefix", action="store_true")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--relength", action="store_true")
    args = parser.parse_args()
    main(args)

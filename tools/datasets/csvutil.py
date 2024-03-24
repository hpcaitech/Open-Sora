import argparse
import csv
import html
import os
import re

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


def shard_csv(data, shard, input_path):
    n = len(data)
    num_shard = n // shard
    for i in range(shard):
        start = i * num_shard
        end = (i + 1) * num_shard if i != shard - 1 else n
        file_name, ext = os.path.splitext(input_path)
        output_path = os.path.join(os.path.join(os.path.dirname(file_name)), f"{os.path.basename(file_name)}_{i}{ext}")
        with open(output_path, "w") as f:
            writer = csv.writer(f)
            writer.writerows(data[start:end])
        print(f"Saved {end - start} samples to {output_path}.")
    print("Done.")


def main(args):
    input_path = args.input
    output_path = args.output
    if output_path is None:
        name = os.path.basename(input_path)
        name, ext = os.path.splitext(name)
        if args.fmin is not None:
            name += f"_fmin_{args.fmin}"
        if args.fmax is not None:
            name += f"_fmax_{args.fmax}"
        if args.remove_empty_caption:
            name += "_rec"
        if args.remove_caption_prefix:
            name += "_rcp"
        if args.root is not None:
            name += f"_root"
        if args.relength:
            name += "_relength"
        if args.clean:
            name += "_clean"
        if args.unescape:
            name += "_unescape"
        if args.remove_url:
            name += "_nourl"
        if args.lang is not None:
            name += f"_{args.lang}"
            from lingua import Language, LanguageDetectorBuilder

            lang_dict = dict(en=Language.ENGLISH)
            assert args.lang in lang_dict
            valid_lang = lang_dict[args.lang]
            detector = LanguageDetectorBuilder.from_all_spoken_languages().with_low_accuracy_mode().build()
        output_path = os.path.join(os.path.dirname(input_path), name + ext)

    with open(input_path, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    print("Number of videos before filtering:", len(data))

    if args.shard is not None:
        shard_csv(data, args.shard, input_path)
        return

    data_new = []
    for i, row in tqdm(enumerate(data)):
        path = row[0]
        caption = row[1]
        if len(row) >= 3:
            n_frames = int(row[2])
        if args.fmin is not None and n_frames < args.fmin:
            continue
        if args.fmax is not None and n_frames > args.fmax:
            continue
        if args.remove_empty_caption and len(caption) == 0:
            continue
        if args.clean:
            caption = caption.strip()
            row[1] = caption
        if args.unescape:
            caption = html.unescape(caption)
            row[1] = caption
        if args.lang is not None:
            confidence_values = detector.compute_language_confidence_values(caption)
            confidence = [x.language for x in confidence_values[:5]]
            if valid_lang not in confidence:
                continue
        if args.remove_url:
            if re.search("(?P<url>https?://[^\s]+)", caption) is not None:
                continue
        if args.remove_caption_prefix:
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
            if len(row) < 3:
                row.append(n_frames)
            else:
                row[2] = n_frames
        data_new.append(row)

    print("Number of videos after filtering:", len(data_new))
    with open(output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data_new)
    print("Output saved to", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fmin", type=int, default=None)
    parser.add_argument("--fmax", type=int, default=None)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--remove-empty-caption", action="store_true")
    parser.add_argument("--remove-caption-prefix", action="store_true")
    parser.add_argument("--relength", action="store_true")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--lang", type=str, default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--unescape", action="store_true")
    parser.add_argument("--remove-url", action="store_true")
    args = parser.parse_args()
    main(args)

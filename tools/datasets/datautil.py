import argparse
import html
import json
import os
import random
import re
from functools import partial
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torchvision
from tqdm import tqdm

from .utils import IMG_EXTENSIONS

tqdm.pandas()

try:
    from pandarallel import pandarallel

    PANDA_USE_PARALLEL = True
except ImportError:
    PANDA_USE_PARALLEL = False


def apply(df, func, **kwargs):
    if PANDA_USE_PARALLEL:
        return df.parallel_apply(func, **kwargs)
    return df.progress_apply(func, **kwargs)


TRAIN_COLUMNS = ["path", "text", "num_frames", "fps", "height", "width", "aspect_ratio", "resolution", "text_len"]

# ======================================================
# --info
# ======================================================


def get_video_length(cap, method="header"):
    assert method in ["header", "set"]
    if method == "header":
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        length = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    return length


def get_info(path):
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in IMG_EXTENSIONS:
            im = cv2.imread(path)
            if im is None:
                return 0, 0, 0, np.nan, np.nan
            height, width = im.shape[:2]
            num_frames, fps = 1, np.nan
        else:
            cap = cv2.VideoCapture(path)
            num_frames, height, width, fps = (
                get_video_length(cap, method="header"),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                float(cap.get(cv2.CAP_PROP_FPS)),
            )
        hw = height * width
        aspect_ratio = height / width if width > 0 else np.nan
        return num_frames, height, width, aspect_ratio, fps, hw
    except:
        return 0, 0, 0, np.nan, np.nan, np.nan


def get_video_info(path):
    try:
        vframes, _, _ = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
        num_frames, height, width = vframes.shape[0], vframes.shape[2], vframes.shape[3]
        aspect_ratio = height / width
        fps = np.nan
        resolution = height * width
        return num_frames, height, width, aspect_ratio, fps, resolution
    except:
        return 0, 0, 0, np.nan, np.nan, np.nan


# ======================================================
# --refine-llm-caption
# ======================================================

LLAVA_PREFIX = [
    "The video shows",
    "The video captures",
    "The video features",
    "The video depicts",
    "The video presents",
    "The video features",
    "The video is ",
    "In the video,",
    "The image shows",
    "The image captures",
    "The image features",
    "The image depicts",
    "The image presents",
    "The image features",
    "The image is ",
    "The image portrays",
    "In the image,",
]


def remove_caption_prefix(caption):
    for prefix in LLAVA_PREFIX:
        if caption.startswith(prefix) or caption.startswith(prefix.lower()):
            caption = caption[len(prefix) :].strip()
            if caption[0].islower():
                caption = caption[0].upper() + caption[1:]
            return caption
    return caption


# ======================================================
# --merge-cmotion
# ======================================================

CMOTION_TEXT = {
    "static": "The camera is static.",
    "dynamic": "The camera is moving.",
    "unknown": None,
    "zoom in": "The camera is zooming in.",
    "zoom out": "The camera is zooming out.",
    "pan left": "The camera is panning left.",
    "pan right": "The camera is panning right.",
    "tilt up": "The camera is tilting up.",
    "tilt down": "The camera is tilting down.",
    "pan/tilt": "The camera is panning.",
}
CMOTION_PROBS = {
    # hard-coded probabilities
    "static": 1.0,
    "dynamic": 1.0,
    "unknown": 0.0,
    "zoom in": 1.0,
    "zoom out": 1.0,
    "pan left": 1.0,
    "pan right": 1.0,
    "tilt up": 1.0,
    "tilt down": 1.0,
    "pan/tilt": 1.0,
}


def merge_cmotion(caption, cmotion):
    text = CMOTION_TEXT[cmotion]
    prob = CMOTION_PROBS[cmotion]
    if text is not None and random.random() < prob:
        caption = f"{caption} {text}"
    return caption


# ======================================================
# --lang
# ======================================================


def build_lang_detector(lang_to_detect):
    from lingua import Language, LanguageDetectorBuilder

    lang_dict = dict(en=Language.ENGLISH)
    assert lang_to_detect in lang_dict
    valid_lang = lang_dict[lang_to_detect]
    detector = LanguageDetectorBuilder.from_all_spoken_languages().with_low_accuracy_mode().build()

    def detect_lang(caption):
        confidence_values = detector.compute_language_confidence_values(caption)
        confidence = [x.language for x in confidence_values[:5]]
        if valid_lang not in confidence:
            return False
        return True

    return detect_lang


# ======================================================
# --clean-caption
# ======================================================


def basic_clean(text):
    import ftfy

    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


BAD_PUNCT_REGEX = re.compile(
    r"[" + "#®•©™&@·º½¾¿¡§~" + "\)" + "\(" + "\]" + "\[" + "\}" + "\{" + "\|" + "\\" + "\/" + "\*" + r"]{1,}"
)  # noqa


def clean_caption(caption):
    import urllib.parse as ul

    from bs4 import BeautifulSoup

    caption = str(caption)
    caption = ul.unquote_plus(caption)
    caption = caption.strip().lower()
    caption = re.sub("<person>", "person", caption)
    # urls:
    caption = re.sub(
        r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
        "",
        caption,
    )  # regex for urls
    caption = re.sub(
        r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
        "",
        caption,
    )  # regex for urls
    # html:
    caption = BeautifulSoup(caption, features="html.parser").text

    # @<nickname>
    caption = re.sub(r"@[\w\d]+\b", "", caption)

    # 31C0—31EF CJK Strokes
    # 31F0—31FF Katakana Phonetic Extensions
    # 3200—32FF Enclosed CJK Letters and Months
    # 3300—33FF CJK Compatibility
    # 3400—4DBF CJK Unified Ideographs Extension A
    # 4DC0—4DFF Yijing Hexagram Symbols
    # 4E00—9FFF CJK Unified Ideographs
    caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
    caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
    caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
    caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
    caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
    caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
    caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
    #######################################################

    # все виды тире / all types of dash --> "-"
    caption = re.sub(
        r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
        "-",
        caption,
    )

    # кавычки к одному стандарту
    caption = re.sub(r"[`´«»“”¨]", '"', caption)
    caption = re.sub(r"[‘’]", "'", caption)

    # &quot;
    caption = re.sub(r"&quot;?", "", caption)
    # &amp
    caption = re.sub(r"&amp", "", caption)

    # ip adresses:
    caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

    # article ids:
    caption = re.sub(r"\d:\d\d\s+$", "", caption)

    # \n
    caption = re.sub(r"\\n", " ", caption)

    # "#123"
    caption = re.sub(r"#\d{1,3}\b", "", caption)
    # "#12345.."
    caption = re.sub(r"#\d{5,}\b", "", caption)
    # "123456.."
    caption = re.sub(r"\b\d{6,}\b", "", caption)
    # filenames:
    caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

    #
    caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
    caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

    caption = re.sub(BAD_PUNCT_REGEX, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
    caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

    # this-is-my-cute-cat / this_is_my_cute_cat
    regex2 = re.compile(r"(?:\-|\_)")
    if len(re.findall(regex2, caption)) > 3:
        caption = re.sub(regex2, " ", caption)

    caption = basic_clean(caption)

    caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
    caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
    caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

    caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
    caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
    caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
    caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
    caption = re.sub(r"\bpage\s+\d+\b", "", caption)

    caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

    caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

    caption = re.sub(r"\b\s+\:\s+", r": ", caption)
    caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
    caption = re.sub(r"\s+", " ", caption)

    caption.strip()

    caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
    caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
    caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
    caption = re.sub(r"^\.\S+$", "", caption)

    return caption.strip()


def text_preprocessing(text, use_text_preprocessing: bool = True):
    if use_text_preprocessing:
        # The exact text cleaning as was in the training stage:
        text = clean_caption(text)
        text = clean_caption(text)
        return text
    else:
        return text.lower().strip()


# ======================================================
# load caption
# ======================================================


def load_caption(path, ext):
    try:
        assert ext in ["json"]
        json_path = path.split(".")[0] + ".json"
        with open(json_path, "r") as f:
            data = json.load(f)
        caption = data["caption"]
        return caption
    except:
        return ""


# ======================================================
# read & write
# ======================================================


def read_file(input_path):
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    elif input_path.endswith(".parquet"):
        return pd.read_parquet(input_path)
    else:
        raise NotImplementedError(f"Unsupported file format: {input_path}")


def save_file(data, output_path):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != "":
        os.makedirs(output_dir)
    if output_path.endswith(".csv"):
        return data.to_csv(output_path, index=False)
    elif output_path.endswith(".parquet"):
        return data.to_parquet(output_path, index=False)
    else:
        raise NotImplementedError(f"Unsupported file format: {output_path}")


def read_data(input_paths):
    data = []
    input_name = ""
    input_list = []
    for input_path in input_paths:
        input_list.extend(glob(input_path))
    print("Input files:", input_list)
    for i, input_path in enumerate(input_list):
        assert os.path.exists(input_path)
        data.append(read_file(input_path))
        input_name += os.path.basename(input_path).split(".")[0]
        if i != len(input_list) - 1:
            input_name += "+"
        print(f"Loaded {len(data[-1])} samples from {input_path}.")
    data = pd.concat(data, ignore_index=True, sort=False)
    print(f"Total number of samples: {len(data)}.")
    return data, input_name


# ======================================================
# main
# ======================================================
# To add a new method, register it in the main, parse_args, and get_output_path functions, and update the doc at /tools/datasets/README.md#documentation


def main(args):
    # reading data
    data, input_name = read_data(args.input)

    # make difference
    if args.difference is not None:
        data_diff = pd.read_csv(args.difference)
        print(f"Difference csv contains {len(data_diff)} samples.")
        data = data[~data["path"].isin(data_diff["path"])]
        input_name += f"-{os.path.basename(args.difference).split('.')[0]}"
        print(f"Filtered number of samples: {len(data)}.")

    # make intersection
    if args.intersection is not None:
        data_new = pd.read_csv(args.intersection)
        print(f"Intersection csv contains {len(data_new)} samples.")
        cols_to_use = data_new.columns.difference(data.columns)
        cols_to_use = cols_to_use.insert(0, "path")
        data = pd.merge(data, data_new[cols_to_use], on="path", how="inner")
        print(f"Intersection number of samples: {len(data)}.")

    # train columns
    if args.train_column:
        all_columns = data.columns
        columns_to_drop = all_columns.difference(TRAIN_COLUMNS)
        data = data.drop(columns=columns_to_drop)

    # get output path
    output_path = get_output_path(args, input_name)

    # preparation
    if args.lang is not None:
        detect_lang = build_lang_detector(args.lang)
    if args.count_num_token == "t5":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("DeepFloyd/t5-v1_1-xxl")

    # IO-related
    if args.load_caption is not None:
        assert "path" in data.columns
        data["text"] = apply(data["path"], load_caption, ext=args.load_caption)
    if args.info:
        info = apply(data["path"], get_info)
        (
            data["num_frames"],
            data["height"],
            data["width"],
            data["aspect_ratio"],
            data["fps"],
            data["resolution"],
        ) = zip(*info)
    if args.video_info:
        info = apply(data["path"], get_video_info)
        (
            data["num_frames"],
            data["height"],
            data["width"],
            data["aspect_ratio"],
            data["fps"],
            data["resolution"],
        ) = zip(*info)
    if args.ext:
        assert "path" in data.columns
        data = data[apply(data["path"], os.path.exists)]

    # filtering
    if args.remove_url:
        assert "text" in data.columns
        data = data[~data["text"].str.contains(r"(?P<url>https?://[^\s]+)", regex=True)]
    if args.lang is not None:
        assert "text" in data.columns
        data = data[data["text"].progress_apply(detect_lang)]  # cannot parallelize
    if args.remove_empty_caption:
        assert "text" in data.columns
        data = data[data["text"].str.len() > 0]
        data = data[~data["text"].isna()]
    if args.remove_path_duplication:
        assert "path" in data.columns
        data = data.drop_duplicates(subset=["path"])

    # processing
    if args.relpath is not None:
        data["path"] = apply(data["path"], lambda x: os.path.relpath(x, args.relpath))
    if args.abspath is not None:
        data["path"] = apply(data["path"], lambda x: os.path.join(args.abspath, x))
    if args.merge_cmotion:
        data["text"] = apply(data, lambda x: merge_cmotion(x["text"], x["cmotion"]), axis=1)
    if args.refine_llm_caption:
        assert "text" in data.columns
        data["text"] = apply(data["text"], remove_caption_prefix)
    if args.clean_caption:
        assert "text" in data.columns
        data["text"] = apply(
            data["text"],
            partial(text_preprocessing, use_text_preprocessing=True),
        )

    if args.count_num_token is not None:
        assert "text" in data.columns
        data["text_len"] = apply(data["text"], lambda x: len(tokenizer(x)["input_ids"]))

    # sort
    if args.sort is not None:
        data = data.sort_values(by=args.sort, ascending=False)
    if args.sort_ascending is not None:
        data = data.sort_values(by=args.sort_ascending, ascending=True)

    # filtering
    if args.remove_empty_caption:
        assert "text" in data.columns
        data = data[data["text"].str.len() > 0]
        data = data[~data["text"].isna()]
    if args.fmin is not None:
        assert "num_frames" in data.columns
        data = data[data["num_frames"] >= args.fmin]
    if args.fmax is not None:
        assert "num_frames" in data.columns
        data = data[data["num_frames"] <= args.fmax]
    if args.hwmax is not None:
        if "resolution" not in data.columns:
            height = data["height"]
            width = data["width"]
            data["resolution"] = height * width
        data = data[data["resolution"] <= args.hwmax]
    if args.aesmin is not None:
        assert "aes" in data.columns
        data = data[data["aes"] >= args.aesmin]
    if args.matchmin is not None:
        assert "match" in data.columns
        data = data[data["match"] >= args.matchmin]
    if args.flowmin is not None:
        assert "flow" in data.columns
        data = data[data["flow"] >= args.flowmin]
    if args.remove_text_duplication:
        data = data.drop_duplicates(subset=["text"], keep="first")
    print(f"Filtered number of samples: {len(data)}.")

    # shard data
    if args.shard is not None:
        sharded_data = np.array_split(data, args.shard)
        for i in range(args.shard):
            output_path_part = output_path.split(".")
            output_path_s = ".".join(output_path_part[:-1]) + f"_{i}." + output_path_part[-1]
            save_file(sharded_data[i], output_path_s)
            print(f"Saved {len(sharded_data[i])} samples to {output_path_s}.")
    else:
        save_file(data, output_path)
        print(f"Saved {len(data)} samples to {output_path}.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, nargs="+", help="path to the input dataset")
    parser.add_argument("--output", type=str, default=None, help="output path")
    parser.add_argument("--format", type=str, default="csv", help="output format", choices=["csv", "parquet"])
    parser.add_argument("--disable-parallel", action="store_true", help="disable parallel processing")
    parser.add_argument("--num-workers", type=int, default=None, help="number of workers")
    parser.add_argument("--seed", type=int, default=None, help="random seed")

    # special case
    parser.add_argument("--shard", type=int, default=None, help="shard the dataset")
    parser.add_argument("--sort", type=str, default=None, help="sort by column")
    parser.add_argument("--sort-ascending", type=str, default=None, help="sort by column (ascending order)")
    parser.add_argument("--difference", type=str, default=None, help="get difference from the dataset")
    parser.add_argument(
        "--intersection", type=str, default=None, help="keep the paths in csv from the dataset and merge columns"
    )
    parser.add_argument("--train-column", action="store_true", help="only keep the train column")

    # IO-related
    parser.add_argument("--info", action="store_true", help="get the basic information of each video and image")
    parser.add_argument("--video-info", action="store_true", help="get the basic information of each video")
    parser.add_argument("--ext", action="store_true", help="check if the file exists")
    parser.add_argument(
        "--load-caption", type=str, default=None, choices=["json", "txt"], help="load the caption from json or txt"
    )

    # path processing
    parser.add_argument("--relpath", type=str, default=None, help="modify the path to relative path by root given")
    parser.add_argument("--abspath", type=str, default=None, help="modify the path to absolute path by root given")

    # caption filtering
    parser.add_argument(
        "--remove-empty-caption",
        action="store_true",
        help="remove rows with empty caption",
    )
    parser.add_argument("--remove-url", action="store_true", help="remove rows with url in caption")
    parser.add_argument("--lang", type=str, default=None, help="remove rows with other language")
    parser.add_argument("--remove-path-duplication", action="store_true", help="remove rows with duplicated path")
    parser.add_argument("--remove-text-duplication", action="store_true", help="remove rows with duplicated caption")

    # caption processing
    parser.add_argument("--refine-llm-caption", action="store_true", help="modify the caption generated by LLM")
    parser.add_argument(
        "--clean-caption", action="store_true", help="modify the caption according to T5 pipeline to suit training"
    )
    parser.add_argument("--merge-cmotion", action="store_true", help="merge the camera motion to the caption")
    parser.add_argument(
        "--count-num-token", type=str, choices=["t5"], default=None, help="Count the number of tokens in the caption"
    )

    # score filtering
    parser.add_argument("--fmin", type=int, default=None, help="filter the dataset by minimum number of frames")
    parser.add_argument("--fmax", type=int, default=None, help="filter the dataset by maximum number of frames")
    parser.add_argument("--hwmax", type=int, default=None, help="filter the dataset by maximum resolution")
    parser.add_argument("--aesmin", type=float, default=None, help="filter the dataset by minimum aes score")
    parser.add_argument("--matchmin", type=float, default=None, help="filter the dataset by minimum match score")
    parser.add_argument("--flowmin", type=float, default=None, help="filter the dataset by minimum flow score")

    return parser.parse_args()


def get_output_path(args, input_name):
    if args.output is not None:
        return args.output
    name = input_name
    dir_path = os.path.dirname(args.input[0])

    # sort
    if args.sort is not None:
        assert args.sort_ascending is None
        name += "_sort"
    if args.sort_ascending is not None:
        assert args.sort is None
        name += "_sort"

    # IO-related
    # for IO-related, the function must be wrapped in try-except
    if args.info:
        name += "_info"
    if args.video_info:
        name += "_vinfo"
    if args.ext:
        name += "_ext"
    if args.load_caption:
        name += f"_load{args.load_caption}"

    # path processing
    if args.relpath is not None:
        name += "_relpath"
    if args.abspath is not None:
        name += "_abspath"

    # caption filtering
    if args.remove_empty_caption:
        name += "_noempty"
    if args.remove_url:
        name += "_nourl"
    if args.lang is not None:
        name += f"_{args.lang}"
    if args.remove_path_duplication:
        name += "_noduppath"
    if args.remove_text_duplication:
        name += "_noduptext"

    # caption processing
    if args.refine_llm_caption:
        name += "_llm"
    if args.clean_caption:
        name += "_clean"
    if args.merge_cmotion:
        name += "_cmcaption"
    if args.count_num_token:
        name += "_ntoken"

    # score filtering
    if args.fmin is not None:
        name += f"_fmin{args.fmin}"
    if args.fmax is not None:
        name += f"_fmax{args.fmax}"
    if args.hwmax is not None:
        name += f"_hwmax{args.hwmax}"
    if args.aesmin is not None:
        name += f"_aesmin{args.aesmin}"
    if args.matchmin is not None:
        name += f"_matchmin{args.matchmin}"
    if args.flowmin is not None:
        name += f"_flowmin{args.flowmin}"

    output_path = os.path.join(dir_path, f"{name}.{args.format}")
    return output_path


if __name__ == "__main__":
    args = parse_args()
    if args.disable_parallel:
        PANDA_USE_PARALLEL = False
    if PANDA_USE_PARALLEL:
        if args.num_workers is not None:
            pandarallel.initialize(nb_workers=args.num_workers, progress_bar=True)
        else:
            pandarallel.initialize(progress_bar=True)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    main(args)

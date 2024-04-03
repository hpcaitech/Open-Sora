import argparse
import html
import os
import re
from functools import partial
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

try:
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True)
    pandas_has_parallel = True
except ImportError:
    pandas_has_parallel = False


def apply(df, func):
    if pandas_has_parallel:
        return df.parallel_apply(func)
    return df.progress_apply(func)


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def get_video_info(path):
    import cv2

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
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            float(cap.get(cv2.CAP_PROP_FPS)),
        )
    hw = height * width
    aspect_ratio = height / width if width > 0 else np.nan
    return num_frames, height, width, aspect_ratio, fps, hw


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
        if isinstance(caption, float):
            breakpoint()
        if caption.startswith(prefix):
            caption = caption[len(prefix) :].strip()
            if caption[0].islower():
                caption = caption[0].upper() + caption[1:]
            return caption


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, nargs="+")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--disable-parallel", action="store_true")
    # special case
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--sort-descending", type=str, default=None)
    parser.add_argument("--sort-ascending", type=str, default=None)
    parser.add_argument("--difference", type=str, default=None)
    parser.add_argument("--intersection", type=str, default=None)

    # path processing
    parser.add_argument("--relpath", type=str, default=None)
    parser.add_argument("--abspath", type=str, default=None)
    # path filtering
    parser.add_argument("--ext", action="store_true")
    # caption filtering
    parser.add_argument("--remove-empty-caption", action="store_true")
    parser.add_argument("--lang", type=str, default=None)
    parser.add_argument("--remove-url", action="store_true")
    # caption processing
    parser.add_argument("--remove-caption-prefix", action="store_true")
    parser.add_argument("--unescape", action="store_true")
    parser.add_argument("--clean-caption", action="store_true")
    # num_frames processing
    parser.add_argument("--info", action="store_true")
    # num_frames filtering
    parser.add_argument("--fmin", type=int, default=None)
    parser.add_argument("--fmax", type=int, default=None)
    # aesthetic filtering
    parser.add_argument("--aesmin", type=float, default=None)
    parser.add_argument("--matchmin", type=float, default=None)

    return parser.parse_args()


def get_output_path(args, input_name):
    if args.output is not None:
        return args.output

    name = input_name
    dir_path = os.path.dirname(args.input[0])

    # path processing
    if args.relpath is not None:
        name += "_relpath"
    if args.abspath is not None:
        name += "_abspath"
    # path filtering
    if args.ext:
        name += "_ext"
    # caption filtering
    if args.remove_empty_caption:
        name += "_noempty"
    if args.lang is not None:
        name += f"_{args.lang}"
    if args.remove_url:
        name += "_nourl"
    # caption processing
    if args.remove_caption_prefix:
        name += "_rcp"
    if args.unescape:
        name += "_unescape"
    if args.clean_caption:
        name += "_clean"
    # num_frames processing
    if args.info:
        name += "_info"
    # num_frames filtering
    if args.fmin is not None:
        name += f"_fmin{args.fmin}"
    if args.fmax is not None:
        name += f"_fmax{args.fmax}"
    # aesthetic filtering
    if args.aesmin is not None:
        name += f"_aesmin{args.aesmin}"
    # clip score filtering
    if args.matchmin is not None:
        name += f"_matchmin{args.matchmin}"
    # sort
    if args.sort_descending is not None:
        assert args.sort_ascending is None
        name += "_sort"
    if args.sort_ascending is not None:
        assert args.sort_descending is None
        name += "_sort"

    output_path = os.path.join(dir_path, f"{name}.csv")
    return output_path


def main(args):
    # reading data
    data = []
    input_name = ""
    input_list = []
    for input_path in args.input:
        input_list.extend(glob(input_path))
    print("Input files:", input_list)
    for i, input_path in enumerate(input_list):
        data.append(pd.read_csv(input_path))
        input_name += os.path.basename(input_path).split(".")[0]
        if i != len(input_list) - 1:
            input_name += "+"
        print(f"Loaded {len(data[-1])} samples from {input_path}.")
    data = pd.concat(data, ignore_index=True, sort=False)
    print(f"Total number of samples: {len(data)}.")

    # make difference
    if args.difference is not None:
        data_diff = pd.read_csv(args.difference)
        print(f"Difference csv contains {len(data_diff)} samples.")
        data = data[~data["path"].isin(data_diff["path"])]
        input_name += f"-{os.path.basename(args.difference).split('.')[0]}"
        print(f"Filtered number of samples: {len(data)}.")

    # make intersection
    if args.intersection is not None:
        data_int = pd.read_csv(args.intersection)
        print(f"Intersection csv contains {len(data_int)} samples.")
        data = data[data["path"].isin(data_int["path"])]
        input_name += f"-{os.path.basename(args.intersection).split('.')[0]}"
        print(f"Filtered number of samples: {len(data)}.")

    # get output path
    output_path = get_output_path(args, input_name)

    # preparation
    if args.lang is not None:
        detect_lang = build_lang_detector(args.lang)

    # filtering
    if args.ext:
        assert "path" in data.columns
        data = data[apply(data["path"], os.path.exists)]
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

    # processing
    if args.relpath is not None:
        data["path"] = apply(data["path"], lambda x: os.path.relpath(x, args.relpath))
    if args.abspath is not None:
        data["path"] = apply(data["path"], lambda x: os.path.join(args.abspath, x))
    if args.remove_caption_prefix:
        assert "text" in data.columns
        data["text"] = apply(data["text"], remove_caption_prefix)
    if args.unescape:
        assert "text" in data.columns
        data["text"] = apply(data["text"], html.unescape)
    if args.clean_caption:
        assert "text" in data.columns
        data["text"] = apply(
            data["text"],
            partial(text_preprocessing, use_text_preprocessing=True),
        )
    if args.info:
        info = apply(data["path"], get_video_info)
        (
            data["num_frames"],
            data["height"],
            data["width"],
            data["aspect_ratio"],
            data["fps"],
            data["resolution"],
        ) = zip(*info)

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
    if args.aesmin is not None:
        assert "aes" in data.columns
        data = data[data["aes"] >= args.aesmin]
    if args.matchmin is not None:
        assert "match" in data.columns
        data = data[data["match"] >= args.matchmin]
    print(f"Filtered number of samples: {len(data)}.")

    # sort
    if args.sort_descending is not None:
        data = data.sort_values(by=args.sort_descending, ascending=False)
    if args.sort_ascending is not None:
        data = data.sort_values(by=args.sort_ascending, ascending=True)

    # shard data
    if args.shard is not None:
        sharded_data = np.array_split(data, args.shard)
        for i in range(args.shard):
            output_path_s = output_path.replace(".csv", f"_{i}.csv")
            sharded_data[i].to_csv(output_path_s, index=False)
            print(f"Saved {len(sharded_data[i])} samples to {output_path_s}.")
    else:
        data.to_csv(output_path, index=False)
        print(f"Saved {len(data)} samples to {output_path}.")


if __name__ == "__main__":
    args = parse_args()
    if args.disable_parallel:
        pandas_has_parallel = False
    main(args)

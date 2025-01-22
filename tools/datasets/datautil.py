import argparse
import html
import json
import math
import os
import random
import re
from functools import partial
from glob import glob

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from opensora.datasets.read_video import read_video

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
PRE_TRAIN_COLUMNS = [
    "path",
    "text",
    "num_frames",
    "fps",
    "height",
    "width",
    "aspect_ratio",
    "resolution",
    "text_len",
    "aes",
    "flow",
    "pred_score",
]

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
            return get_image_info(path)
        else:
            return get_video_info(path)
    except:
        return 0, 0, 0, np.nan, np.nan, np.nan


def get_image_info(path, backend="pillow"):
    if backend == "pillow":
        try:
            with open(path, "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")
            width, height = img.size
            num_frames, fps = 1, np.nan
            hw = height * width
            aspect_ratio = height / width if width > 0 else np.nan
            return num_frames, height, width, aspect_ratio, fps, hw
        except:
            return 0, 0, 0, np.nan, np.nan, np.nan
    elif backend == "cv2":
        try:
            im = cv2.imread(path)
            if im is None:
                return 0, 0, 0, np.nan, np.nan, np.nan
            height, width = im.shape[:2]
            num_frames, fps = 1, np.nan
            hw = height * width
            aspect_ratio = height / width if width > 0 else np.nan
            return num_frames, height, width, aspect_ratio, fps, hw
        except:
            return 0, 0, 0, np.nan, np.nan, np.nan
    else:
        raise ValueError


def get_video_info(path, backend="torchvision"):
    if backend == "torchvision":
        try:
            vframes, infos = read_video(path)
            num_frames, height, width = vframes.shape[0], vframes.shape[2], vframes.shape[3]
            if "video_fps" in infos:
                fps = infos["video_fps"]
            else:
                fps = np.nan
            hw = height * width
            aspect_ratio = height / width if width > 0 else np.nan
            return num_frames, height, width, aspect_ratio, fps, hw
        except:
            return 0, 0, 0, np.nan, np.nan, np.nan
    elif backend == "cv2":
        try:
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
    else:
        raise ValueError


# ======================================================
# --refine-llm-caption
# ======================================================

LLAVA_PREFIX = [
    "The video shows ",
    "The video captures ",
    "The video features ",
    "The video depicts ",
    "The video presents ",
    "The video features ",
    "The video is ",
    "In the video, ",
    "The image shows ",
    "The image captures ",
    "The image features ",
    "The image depicts ",
    "The image presents ",
    "The image features ",
    "The image is ",
    "The image portrays ",
    "In the image, ",
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
    "static": "static",
    "pan_right": "pan right",
    "pan_left": "pan left",
    "zoom_in": "zoom in",
    "zoom_out": "zoom out",
    "tilt_up": "tilt up",
    "tilt_down": "tilt down",
    # "pan/tilt": "The camera is panning.",
    # "dynamic": "The camera is moving.",
    # "unknown": None,
}
CMOTION_PROBS = {
    # hard-coded probabilities
    "static": 1.0,
    "zoom_in": 1.0,
    "zoom_out": 1.0,
    "pan_left": 1.0,
    "pan_right": 1.0,
    "tilt_up": 1.0,
    "tilt_down": 1.0,
    # "dynamic": 1.0,
    # "unknown": 0.0,
    # "pan/tilt": 1.0,
}


def merge_cmotion(caption, cmotion):
    text = CMOTION_TEXT[cmotion]
    prob = CMOTION_PROBS[cmotion]
    if text is not None and random.random() < prob:
        caption = f"{caption} Camera motion: {text}."
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


def text_refine_t5(caption):
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
        text = text_refine_t5(text)
        text = text_refine_t5(text)
        return text
    else:
        return text.lower().strip()


def has_human(text):
    first_sentence = text.split(".")[0]
    human_words = ["man", "woman", "child", "girl", "boy"]
    for word in human_words:
        if word in first_sentence:
            return True
    return False


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
# --clean-caption
# ======================================================

DROP_SCORE_PROB = 0.2


def transform_aes(aes):
    # < 4 filter out
    if aes < 4:
        return "terrible"
    elif aes < 4.5:
        return "very poor"
    elif aes < 5:
        return "poor"
    elif aes < 5.5:
        return "fair"
    elif aes < 6:
        return "good"
    elif aes < 6.5:
        return "very good"
    else:
        return "excellent"


def transform_motion(motion):
    # < 0.3 filter out
    if motion < 0.5:
        return "very low"
    elif motion < 2:
        return "low"
    elif motion < 5:
        return "fair"
    elif motion < 10:
        return "high"
    elif motion < 20:
        return "very high"
    else:
        return "extremely high"


def score2text(data):
    text = data["text"]
    if not text.endswith("."):
        text += "."
    # aesthetic
    if "aes" in data:
        aes = transform_aes(data["aes"])
        if random.random() > DROP_SCORE_PROB:
            score_text = f" the aesthetic score is {aes}."
            text += score_text
    # flow
    if "flow" in data:
        flow = transform_motion(data["flow"])
        if random.random() > DROP_SCORE_PROB:
            score_text = f" the motion strength is {flow}."
            text += score_text
    return text


def undo_score2text(data):
    text = data["text"]
    sentences = text.strip().split(".")[:-1]

    keywords = ["aesthetic score", "motion strength"]
    num_scores = len(keywords)
    num_texts_from_score = 0
    for idx in range(1, num_scores + 1):
        s = sentences[-idx]

        for key in keywords:
            if key in s:
                num_texts_from_score += 1
                break

    new_sentences = sentences[:-num_texts_from_score] if num_texts_from_score > 0 else sentences
    new_text = ".".join(new_sentences)
    if not new_text.endswith("."):
        new_text += "."
    return new_text


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
    if len(input_paths) == 0:
        print(f"No meta file to process. Exit.")
        exit()

    data = []
    input_name = ""
    input_list = []
    for input_path in input_paths:
        input_list.extend(glob(input_path))
    cnt = len(input_list)
    print(f"==> Total {cnt} input files:")
    for x in input_list:
        print(x)

    for i, input_path in enumerate(input_list):
        if not os.path.exists(input_path):
            raise FileNotFoundError
        data.append(read_file(input_path))
        basename = os.path.basename(input_path)
        input_name += os.path.splitext(basename)[0]
        if i != len(input_list) - 1:
            input_name += "+"
        print(f"==> Loaded meta (shape={data[-1].shape}) from '{input_path}'")

    data = pd.concat(data, ignore_index=True, sort=False)
    print(f"==> Merged {cnt} files. shape={data.shape}")
    return data, input_name


def is_verbose_sentence(s):
    if ("is no " in s) or ("are no " in s):
        return True
    if "does not " in s:
        return True
    if "solely " in s:
        return True
    if "only " in s:
        return True
    if "not visible" in s:
        return True
    if "no " in s and "visible" in s:
        return True
    return False


def is_verbose_caption(caption):
    caption = caption.strip()
    sentences = caption.split(".")
    if not caption.endswith("."):
        sentences = sentences[:-1]

    cnt = 0
    for sentence in sentences:
        if is_verbose_sentence(sentence):
            cnt += 1
        if cnt >= 2:
            return True
    return False


def refine_sentences(caption):
    caption = caption.strip()
    sentences = caption.split(".")
    if not caption.endswith("."):
        sentences = sentences[:-1]

    new_caption = ""
    for i, sentence in enumerate(sentences):
        if sentence.strip() == "":
            continue
        if is_verbose_sentence(sentence):
            continue
        new_caption += f"{sentence}."
    return new_caption


# ======================================================
# main
# ======================================================
# To add a new method, register it in the main, parse_args, and get_output_path functions, and update the doc at /tools/datasets/README.md#documentation


def main(args):
    # reading data
    data, input_name = read_data(args.input)

    # get output path
    output_path = get_output_path(args, input_name)

    # path subtract (difference set)
    if args.path_subtract is not None:
        data_diff = pd.read_csv(args.path_subtract)
        print(f"Meta to subtract: shape={data_diff.shape}.")
        data = data[~data["path"].isin(data_diff["path"])]

    # path intersect
    if args.path_intersect is not None:
        data_new = pd.read_csv(args.path_intersect)
        print(f"Meta to intersect: shape={data_new.shape}.")

        new_cols = data_new.columns.difference(data.columns)
        col_on = "path"
        new_cols = new_cols.insert(0, col_on)
        data = pd.merge(data, data_new[new_cols], on=col_on, how="inner")

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

    # filtering path
    if args.path_filter_empty:
        assert "path" in data.columns
        data = data[data["path"].str.len() > 0]
        data = data[~data["path"].isna()]
    if args.path_filter_substr:
        data = data[~data["path"].str.contains(args.path_filter_substr)]
    if args.path_keep_substr:
        data = data[data["path"].str.contains(args.path_keep_substr)]
    if args.path_dedup:
        assert "path" in data.columns
        data = data.drop_duplicates(subset=["path"])

    # filtering text
    if args.text_filter_url:
        assert "text" in data.columns
        data = data[~data["text"].str.contains(r"(?P<url>https?://[^\s]+)", regex=True)]
    if args.lang is not None:
        assert "text" in data.columns
        data = data[data["text"].progress_apply(detect_lang)]  # cannot parallelize
    if args.text_filter_empty:
        assert "text" in data.columns
        data = data[data["text"].str.len() > 0]
        data = data[~data["text"].isna()]
    if args.text_filter_substr:
        assert "text" in data.columns
        data = data[~data["text"].str.contains(args.text_filter_substr)]

    # processing
    if args.relpath is not None:
        data["path"] = apply(data["path"], lambda x: os.path.relpath(x, args.relpath))
    if args.abspath is not None:
        data["path"] = apply(data["path"], lambda x: os.path.join(args.abspath, x))
    if args.path_to_id:
        data["id"] = apply(data["path"], lambda x: os.path.splitext(os.path.basename(x))[0])
    if args.merge_cmotion:
        data["text"] = apply(data, lambda x: merge_cmotion(x["text"], x["cmotion"]), axis=1)
    if args.text_remove_prefix:
        assert "text" in data.columns
        data["text"] = apply(data["text"], remove_caption_prefix)
    if args.text_append is not None:
        assert "text" in data.columns
        data["text"] = data["text"] + args.text_append
    if args.text_refine_t5:
        assert "text" in data.columns
        data["text"] = apply(
            data["text"],
            partial(text_preprocessing, use_text_preprocessing=True),
        )
    if args.text_image2video:
        assert "text" in data.columns
        data["text"] = apply(data["text"], lambda x: x.replace("still image", "video").replace("image", "video"))
    if args.count_num_token is not None:
        assert "text" in data.columns
        data["text_len"] = apply(data["text"], lambda x: len(tokenizer(x)["input_ids"]))
    if args.update_text is not None:
        data_new = pd.read_csv(args.update_text)
        num_updated = data.path.isin(data_new.path).sum()
        print(f"Number of updated samples: {num_updated}.")
        data = data.set_index("path")
        data_new = data_new[["path", "text"]].set_index("path")
        data.update(data_new)
        data = data.reset_index()
    if args.text_refine_sentences:
        data["text"] = apply(data["text"], refine_sentences)
    if args.text_score2text:
        data["text"] = apply(data, score2text, axis=1)
    if args.text_undo_score2text:
        data["text"] = apply(data, undo_score2text, axis=1)

    # sort
    if args.sort is not None:
        data = data.sort_values(by=args.sort, ascending=False)
    if args.sort_ascending is not None:
        data = data.sort_values(by=args.sort_ascending, ascending=True)

    # filtering
    if args.filesize:
        assert "path" in data.columns
        data["filesize"] = apply(data["path"], lambda x: os.stat(x).st_size / 1024 / 1024)
    if args.fsmax is not None:
        assert "filesize" in data.columns
        data = data[data["filesize"] <= args.fsmax]
    if args.fsmin is not None:
        assert "filesize" in data.columns
        data = data[data["filesize"] >= args.fsmin]
    if args.text_filter_empty:
        assert "text" in data.columns
        data = data[data["text"].str.len() > 0]
        data = data[~data["text"].isna()]
    if args.fmin is not None:
        assert "num_frames" in data.columns
        data = data[data["num_frames"] >= args.fmin]
    if args.fmax is not None:
        assert "num_frames" in data.columns
        data = data[data["num_frames"] <= args.fmax]
    if args.filter_dyn_fps is not False:
        assert "fps" in data.columns and "num_frames" in data.columns

        def dyn_fps_filter(row, max_fps=args.dyn_fps_max_fps, keep_frames=args.dyn_fps_keep_frames):
            # get scale factor
            if math.isnan(row["fps"]):  # image
                return True
            scale_factor = math.ceil(row["fps"] / max_fps)
            min_frames = keep_frames * scale_factor
            return row["num_frames"] >= min_frames

        dyn_fps_mask = data.apply(dyn_fps_filter, axis=1)
        data = data[dyn_fps_mask]
    if args.fpsmax is not None:
        assert "fps" in data.columns
        data = data[(data["fps"] <= args.fpsmax) | np.isnan(data["fps"])]
    if args.hwmax is not None:
        if "resolution" not in data.columns:
            height = data["height"]
            width = data["width"]
            data["resolution"] = height * width
        data = data[data["resolution"] <= args.hwmax]
    if args.aesmin is not None:
        assert "aes" in data.columns
        data = data[data["aes"] >= args.aesmin]
    if args.prefmin is not None:
        assert "pred_score" in data.columns
        data = data[data["pred_score"] >= args.prefmin]
    if args.matchmin is not None:
        assert "match" in data.columns
        data = data[data["match"] >= args.matchmin]
    if args.flowmin is not None:
        assert "flow" in data.columns
        data = data[data["flow"] >= args.flowmin]
    if args.facemin is not None:
        assert "face_area_ratio" in data.columns
        data = data[data["face_area_ratio"] >= args.facemin]
    if args.text_dedup:
        data = data.drop_duplicates(subset=["text"], keep="first")
    if args.img_only:
        data = data[data["path"].str.lower().str.endswith(IMG_EXTENSIONS)]
    if args.vid_only:
        data = data[~data["path"].str.lower().str.endswith(IMG_EXTENSIONS)]
    if args.filter_too_verbose:
        data = data[data["text"].apply(is_verbose_caption)]
    if args.h_le_w:
        data = data[data["height"] <= data["width"]]
    if args.filter_human:
        data = data[data["text"].apply(has_human)]

    if args.ext:
        assert "path" in data.columns
        data = data[apply(data["path"], os.path.exists)]

    # process data
    if args.shuffle:
        data = data.sample(frac=1).reset_index(drop=True)  # shuffle
    if args.head is not None:
        data = data.head(args.head)
    if args.sample is not None:
        data = data.sample(args.sample).reset_index(drop=True)

    # train columns
    if args.train_column:
        assert args.pre_train_column is False
        all_columns = data.columns
        columns_to_drop = all_columns.difference(TRAIN_COLUMNS)
        data = data.drop(columns=columns_to_drop)
    elif args.pre_train_column:
        assert args.train_column is False
        all_columns = data.columns
        columns_to_drop = all_columns.difference(PRE_TRAIN_COLUMNS)
        data = data.drop(columns=columns_to_drop)

    if args.chunk is not None:
        assert len(args.input) == 1
        input_path = args.input[0]
        res = np.array_split(data, args.chunk)
        for idx in range(args.chunk):
            out_path = f"_chunk-{idx}-{args.chunk}".join(os.path.splitext(input_path))
            shape = res[idx].shape
            print(f"==> Saving meta file (shape={shape}) to '{out_path}'")
            if args.format == "csv":
                res[idx].to_csv(out_path, index=False)
            elif args.format == "parquet":
                res[idx].to_parquet(out_path, index=False)
            else:
                raise NotImplementedError
            print(f"New meta (shape={shape}) saved to '{out_path}'")
    else:
        shape = data.shape
        print(f"==> Saving meta file (shape={shape}) to '{output_path}'")
        save_file(data, output_path)
        print(f"==> New meta (shape={shape}) saved to '{output_path}'")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, nargs="+", help="path to the input dataset")
    parser.add_argument("--output", type=str, default=None, help="output path")
    parser.add_argument("--format", type=str, default="csv", help="output format", choices=["csv", "parquet"])
    parser.add_argument("--disable_parallel", action="store_true", help="disable parallel processing")
    parser.add_argument("--num-workers", type=int, default=None, help="number of workers")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    # special case
    parser.add_argument("--shard", type=int, default=None, help="shard the dataset")
    parser.add_argument("--sort", type=str, default=None, help="sort by column")
    parser.add_argument("--sort-ascending", type=str, default=None, help="sort by column (ascending order)")
    parser.add_argument("--path_subtract", type=str, default=None, help="substract path (difference set)")
    parser.add_argument("--path_intersect", type=str, default=None, help="intersect path and merge columns")
    parser.add_argument("--train_column", action="store_true", help="only keep the train column")
    parser.add_argument("--pre_train_column", action="store_true", help="only keep the pre-train column")

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
    parser.add_argument("--path-to-id", action="store_true", help="add id based on path")
    parser.add_argument(
        "--path_filter_empty", action="store_true", help="remove rows with empty path"
    )  # caused by transform, cannot read path
    parser.add_argument(
        "--path_filter_substr", type=str, default=None, help="remove rows whose path contains a substring"
    )
    parser.add_argument("--path_keep_substr", type=str, default=None, help="keep rows whose path contains a substring")
    parser.add_argument("--path_dedup", action="store_true", help="remove rows with duplicated path")

    # caption filtering
    parser.add_argument("--text_filter_empty", action="store_true", help="remove rows with empty caption")
    parser.add_argument("--text_filter_url", action="store_true", help="remove rows with url in caption")
    parser.add_argument("--text_filter_substr", type=str, default=None, help="remove text with a substring")
    parser.add_argument("--text_dedup", action="store_true", help="remove rows with duplicated caption")
    parser.add_argument("--lang", type=str, default=None, help="remove rows with other language")
    parser.add_argument("--filter-too-verbose", action="store_true", help="filter samples with too verbose caption")
    parser.add_argument("--filter-human", action="store_true", help="filter samples with human preference score")

    # caption processing
    parser.add_argument("--text_remove_prefix", action="store_true", help="remove prefix like 'The video shows '")
    parser.add_argument(
        "--text_refine_t5", action="store_true", help="refine the caption output by T5 with regular expression"
    )
    parser.add_argument("--text_image2video", action="store_true", help="text.replace('image', 'video'")
    parser.add_argument("--text_refine_sentences", action="store_true", help="refine every sentence in the caption")
    parser.add_argument("--text_score2text", action="store_true", help="convert score to text and append to caption")
    parser.add_argument("--text_undo_score2text", action="store_true", help="undo score2text")
    parser.add_argument("--merge-cmotion", action="store_true", help="merge the camera motion to the caption")
    parser.add_argument(
        "--count-num-token", type=str, choices=["t5"], default=None, help="Count the number of tokens in the caption"
    )
    parser.add_argument("--text_append", type=str, default=None, help="append text to the caption")
    parser.add_argument("--update-text", type=str, default=None, help="update the text with the given text")

    # filter for dynamic fps
    parser.add_argument(
        "--filter_dyn_fps", action="store_true", help="filter data to contain enough frames for dynamic fps"
    )
    parser.add_argument("--dyn_fps_max_fps", type=int, default=16, help="max fps for dynamic fps")
    parser.add_argument("--dyn_fps_keep_frames", type=int, default=32, help="num frames to keep for dynamic fps")

    # score filtering
    parser.add_argument("--filesize", action="store_true", help="get the filesize of each video and image in MB")
    parser.add_argument("--fsmax", type=float, default=None, help="filter the dataset by maximum filesize")
    parser.add_argument("--fsmin", type=float, default=None, help="filter the dataset by minimum filesize")
    parser.add_argument("--fmin", type=int, default=None, help="filter the dataset by minimum number of frames")
    parser.add_argument("--fmax", type=int, default=None, help="filter the dataset by maximum number of frames")
    parser.add_argument("--hwmax", type=int, default=None, help="filter the dataset by maximum resolution")
    parser.add_argument("--aesmin", type=float, default=None, help="filter the dataset by minimum aes score")
    parser.add_argument(
        "--prefmin", type=float, default=None, help="filter the dataset by minimum human preference score"
    )
    parser.add_argument("--matchmin", type=float, default=None, help="filter the dataset by minimum match score")
    parser.add_argument("--flowmin", type=float, default=None, help="filter the dataset by minimum flow score")
    parser.add_argument("--facemin", type=float, default=None, help="filter the dataset by minimum face area ratio")
    parser.add_argument("--fpsmax", type=float, default=None, help="filter the dataset by maximum fps")
    parser.add_argument("--img-only", action="store_true", help="only keep the image data")
    parser.add_argument("--vid-only", action="store_true", help="only keep the video data")
    parser.add_argument("--h-le-w", action="store_true", help="only keep samples with h <= w")

    # data processing
    parser.add_argument("--shuffle", default=False, action="store_true", help="shuffle the dataset")
    parser.add_argument("--head", type=int, default=None, help="return the first n rows of data")
    parser.add_argument("--sample", type=int, default=None, help="randomly sample n rows; using args.seed")
    parser.add_argument("--chunk", type=int, default=None, help="evenly split rows into chunks")

    return parser.parse_args()


def get_output_path(args, input_name):
    if args.output is not None:
        return args.output
    name = input_name
    dir_path = os.path.dirname(args.input[0])

    if args.path_subtract is not None:
        name += "_subtract"
    if args.path_intersect is not None:
        name += "_intersect"

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
    if args.path_filter_empty:
        name += "_path-filter-empty"
    if args.path_filter_substr is not None:
        name += "_path-filter-substr"
    if args.path_keep_substr is not None:
        name += "_path-keep-substr"
    if args.path_dedup:
        name += "_path-dedup"

    # caption filtering
    if args.text_filter_empty:
        name += "_text-filter-empty"
    if args.text_filter_url:
        name += "_text-filter-url"
    if args.lang is not None:
        name += f"_{args.lang}"
    if args.text_dedup:
        name += "_text-dedup"
    if args.text_filter_substr is not None:
        name += f"_text-filter-substr"
    if args.filter_too_verbose:
        name += "_noverbose"
    if args.h_le_w:
        name += "_h-le-w"
    if args.filter_human:
        name += "_human"

    # caption processing
    if args.text_remove_prefix:
        name += "_text-remove-prefix"
    if args.text_refine_t5:
        name += "_text-refine-t5"
    if args.text_image2video:
        name += "_text-image2video"
    if args.text_refine_sentences:
        name += "_text-refine-sentences"
    if args.text_score2text:
        name += "_text-score2text"
    if args.text_undo_score2text:
        name += "_text-undo-score2text"
    if args.merge_cmotion:
        name += "_cmcaption"
    if args.count_num_token:
        name += "_ntoken"
    if args.text_append is not None:
        name += "_text-append"
    if args.update_text is not None:
        name += "_update"

    # dynmaic fps
    if args.filter_dyn_fps is not False:
        name += "_dynfps"

    # score filtering
    if args.filesize:
        name += "_filesize"
    if args.fsmax is not None:
        name += f"_fsmax{args.fsmax}"
    if args.fsmin is not None:
        name += f"_fsmin{args.fsmin}"
    if args.fmin is not None:
        name += f"_fmin{args.fmin}"
    if args.fmax is not None:
        name += f"_fmax{args.fmax}"
    if args.fpsmax is not None:
        name += f"_fpsmax{args.fpsmax}"
    if args.hwmax is not None:
        name += f"_hwmax{args.hwmax}"
    if args.aesmin is not None:
        name += f"_aesmin{args.aesmin}"
    if args.prefmin is not None:
        name += f"_prefmin{args.prefmin}"
    if args.matchmin is not None:
        name += f"_matchmin{args.matchmin}"
    if args.flowmin is not None:
        name += f"_flowmin{args.flowmin}"
    if args.facemin is not None:
        name += f"_facemin{args.facewmin}"
    if args.img_only:
        name += "_img"
    if args.vid_only:
        name += "_vid"

    # processing
    if args.shuffle:
        name += f"_shuffled_seed{args.seed}"
    if args.head is not None:
        name += f"_first_{args.head}_data"
    if args.sample is not None:
        name += f"_sample-{args.sample}"

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

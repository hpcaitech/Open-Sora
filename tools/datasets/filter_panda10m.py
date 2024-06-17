# TODO: remove this file before releasing

import argparse
import html
import os
import re

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

try:
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True)
    pandas_has_parallel = True
except ImportError:
    pandas_has_parallel = False


def apply(df, func, **kwargs):
    if pandas_has_parallel:
        return df.parallel_apply(func, **kwargs)
    return df.progress_apply(func, **kwargs)


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


def get_10m_set():
    meta_path_10m = "/mnt/hdd/data/Panda-70M/raw/meta/train/panda70m_training_10m.csv"
    meta_10m = pd.read_csv(meta_path_10m)

    def process_single_caption(row):
        text_list = eval(row["caption"])
        clean_list = [clean_caption(x) for x in text_list]
        return str(clean_list)

    ret = apply(meta_10m, process_single_caption, axis=1)
    # ret = meta_10m.progress_apply(process_single_caption, axis=1)
    print("==> text processed.")

    text_list = []
    for x in ret:
        text_list += eval(x)
        # text_set = text_set.union(set(eval(x)))
    text_set = set(text_list)
    # meta_10m['caption_new'] = ret
    # meta_10m.to_csv('/mnt/hdd/data/Panda-70M/raw/meta/train/panda70m_training_10m_new-cap.csv')

    # video_id_set = set(meta_10m['videoID'])
    # id2t = {}
    # for idx, row in tqdm(meta_10m.iterrows(), total=len(meta_10m)):
    #     video_id = row['videoID']
    #     text_list = eval(row['caption'])
    #     id2t[video_id] = set(text_list)

    print(f"==> Loaded meta_10m from '{meta_path_10m}'")
    return text_set


def filter_panda10m_text(meta_path, text_set):
    def process_single_row(row):
        # path = row['path']
        t = row["text"]
        # fname = os.path.basename(path)
        # video_id = fname[:fname.rindex('_')]
        if t not in text_set:
            return False
        return True

    meta = pd.read_csv(meta_path)
    ret = apply(meta, process_single_row, axis=1)
    # ret = meta.progress_apply(process_single_row, axis=1)

    meta = meta[ret]
    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_filter-10m{ext}"
    meta.to_csv(out_path, index=False)
    print(f"New meta (shape={meta.shape}) saved to '{out_path}'.")


def filter_panda10m_timestamp(meta_path):
    meta_path_10m = "/mnt/hdd/data/Panda-70M/raw/meta/train/panda70m_training_10m.csv"
    meta_10m = pd.read_csv(meta_path_10m)

    id2t = {}
    for idx, row in tqdm(meta_10m.iterrows(), total=len(meta_10m)):
        video_id = row["videoID"]
        timestamp = eval(row["timestamp"])
        timestamp = [str(tuple(x)) for x in timestamp]
        id2t[video_id] = timestamp

    # video_id_set_10m = set(meta_10m['videoID'])
    print(f"==> Loaded meta_10m from '{meta_path_10m}'")

    def process_single_row(row):
        path = row["path"]
        t = row["timestamp"]
        fname = os.path.basename(path)
        video_id = fname[: fname.rindex("_")]
        if video_id not in id2t:
            return False
        if t not in id2t[video_id]:
            return False
        return True
        # return video_id in video_id_set_10m

    meta = pd.read_csv(meta_path)
    ret = apply(meta, process_single_row, axis=1)

    meta = meta[ret]
    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_filter-10m{ext}"
    meta.to_csv(out_path, index=False)
    print(f"New meta (shape={meta.shape}) saved to '{out_path}'.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", type=str, nargs="+")
    parser.add_argument("--num_workers", default=5, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    text_set = get_10m_set()
    for x in args.meta_path:
        filter_panda10m_text(x, text_set)

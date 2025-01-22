import argparse
import base64
import csv
import os
from io import BytesIO

import tqdm
from openai import OpenAI

from .utils import IMG_EXTENSIONS, PROMPTS, VID_EXTENSIONS, VideoTextDataset

client = OpenAI()


def to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_caption(text, prompt):
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        max_tokens=300,
        top_p=0.1,
    )
    caption = response.choices[0].message.content
    caption = caption.replace("\n", " ")

    return caption


def main(args):
    # ======================================================
    # 1. read video list
    # ======================================================
    dataset = VideoTextDataset(args.input, text_only=True)
    output_file = os.path.splitext(args.input)[0] + "_captrans.csv"
    f = open(output_file, "w")
    writer = csv.writer(f)
    writer.writerow(["video", "text"])

    # make sure that the prompt type matches the data type
    data_extension = "." + dataset.data["path"].iloc[0].split(".")[-1]
    prompt_type = PROMPTS[args.prompt]["type"]
    if prompt_type == "image":
        assert (
            data_extension.lower() in IMG_EXTENSIONS
        ), "The prompt is suitable for an image dataset but the data is not image."
    elif prompt_type == "video":
        assert (
            data_extension.lower() in VID_EXTENSIONS
        ), "The prompt is suitable for a video dataset but the data is not video."
    else:
        raise ValueError(f"Found invalid prompt type {prompt_type}")

    # ======================================================
    # 2. generate captions
    # ======================================================
    for sample in tqdm.tqdm(dataset):
        prompt = PROMPTS[args.prompt]["text"]
        text = sample["text"]
        caption = get_caption(text, prompt)

        writer.writerow((sample["path"], caption))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to the input CSV file")
    parser.add_argument("--prompt", type=str, default="video-captrans-template")  # 1k/20min
    args = parser.parse_args()

    main(args)

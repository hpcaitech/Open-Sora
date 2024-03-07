import argparse
import math
import os

import torch
from datasets import load_dataset
from torchvision.io import read_video
from transformers import AutoTokenizer, CLIPTextModel

EMPTY_SAMPLE = {"video_file": [], "text_latent_states": []}


def process_text(text, tokenizer, text_model, use_pooled_text):
    inputs = tokenizer(text, padding=True, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = text_model(**inputs)
    if use_pooled_text:
        return list(outputs.pooler_output.cpu().unbind(0))
    output_states = []
    for i, x in enumerate(outputs.last_hidden_state):
        valid_x = x[inputs["attention_mask"][i].bool()]
        output_states.append(valid_x.cpu())
    return output_states


@torch.no_grad()
def process_item(item, video_dir, tokenizer, text_model, use_pooled_text):
    video_path = os.path.join(video_dir, item["file"])
    video = read_video(video_path, pts_unit="sec")[0]
    if video.size(0) > 600:
        return EMPTY_SAMPLE
    text_latent_states = process_text(
        item["captions"], tokenizer, text_model, use_pooled_text
    )
    torch.cuda.empty_cache()
    return {
        "video_file": [item["file"]] * len(text_latent_states),
        "text_latent_states": text_latent_states,
    }


def process_batch(batch, video_dir, tokenizer, text_model, use_pooled_text):
    item = {"file": batch["file"][0], "captions": batch["captions"][0]}
    return process_item(item, video_dir, tokenizer, text_model, use_pooled_text)


def process_dataset(
    captions_file,
    video_dir,
    output_dir,
    num_spliced_dataset_bins=10,
    text_model="openai/clip-vit-base-patch32",
    use_pooled_text=False,
):
    tokenizer = AutoTokenizer.from_pretrained(text_model)
    text_model = CLIPTextModel.from_pretrained(text_model).cuda().eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare to data splitting.
    train_splits = []
    split_interval = math.ceil(100 / num_spliced_dataset_bins)
    for i in range(0, 100, split_interval):
        start = i
        end = i + split_interval
        if end > 100:
            end = 100
        train_splits.append(f"train[{start}%:{end}%]")

    ds = load_dataset(
        "json",
        data_files=captions_file,
        keep_in_memory=False,
        split=train_splits,
        num_proc=1,
    )

    for i, part_ds in enumerate(ds):
        print(f"Processing part {i+1}/{len(ds)}")
        part_ds = part_ds.with_format("torch")
        part_ds = part_ds.map(
            process_batch,
            fn_kwargs={
                "video_dir": video_dir,
                "tokenizer": tokenizer,
                "text_model": text_model,
                "use_pooled_text": use_pooled_text,
            },
            batched=True,
            batch_size=1,
            keep_in_memory=False,
            remove_columns=part_ds.column_names,
        )
        output_path = os.path.join(output_dir, f"part-{i:05d}")
        part_ds.save_to_disk(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument(
        "-c",
        "--captions-file",
        type=str,
        help="Path to the captions file. It should be a JSON file or a JSONL file",
    )
    parser.add_argument(
        "-v", "--video-dir", type=str, help="Path to the video directory"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, help="Path to the output directory"
    )
    parser.add_argument(
        "-n",
        "--num_spliced_dataset_bins",
        type=int,
        default=10,
        help="Number of bins for spliced dataset",
    )
    parser.add_argument(
        "--text_model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP text model",
    )
    parser.add_argument("--use_pooled_text", action="store_true", default=False)
    args = parser.parse_args()
    process_dataset(
        args.captions_file,
        args.video_dir,
        args.output_dir,
        args.num_spliced_dataset_bins,
        args.text_model,
        args.use_pooled_text,
    )

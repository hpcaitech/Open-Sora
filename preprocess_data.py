import argparse
import math
import os

import torch
from datasets import load_dataset
from torchvision.io import read_video
from transformers import AutoModel, AutoTokenizer, CLIPTextModel

EMPTY_SAMPLE = {"video_file": [], "video_latent_states": [], "text_latent_states": []}

def preprocess_video(video):
    # [T, H, W, C] to [C, T, H, W]
    video = video.permute(3, 0, 1, 2)
    video = video.to(dtype=torch.float, device="cuda")
    # normalize
    video = video / 255 - 0.5
    return video.unsqueeze(0)

def process_video(video_path, vqvae):
    video = read_video(video_path, pts_unit="sec")[0]
    video = preprocess_video(video)
    if video.size(2) > 600:
        raise ValueError("Video is too long")
    latent_states = vqvae.encode(video)
    return latent_states.squeeze(0).tolist()

def process_text(text, tokenizer, text_model):
    inputs = tokenizer(text, padding=True, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = text_model(**inputs)
    output_states = []
    for i, x in enumerate(outputs.last_hidden_state):
        valid_x = x[inputs["attention_mask"][i].bool()]
        output_states.append(valid_x.tolist())
    return output_states

@torch.no_grad()
def process_item(item, video_dir, tokenizer, text_model, vqvae):
    video_path = os.path.join(video_dir, item["file"])
    try:
        video_latent_states = process_video(video_path, vqvae)
    except ValueError as e:
        return EMPTY_SAMPLE
    torch.cuda.empty_cache()
    text_latent_states = process_text(item["captions"], tokenizer, text_model)
    torch.cuda.empty_cache()
    return {
        "video_file": [item["file"]] * len(text_latent_states),
        "video_latent_states": [video_latent_states] * len(text_latent_states),
        "text_latent_states": text_latent_states
    }

def process_batch(batch, video_dir, tokenizer, text_model, vqvae):
    item = {"file": batch["file"][0], "captions": batch["captions"][0]}
    return process_item(item, video_dir, tokenizer, text_model, vqvae)

def process_dataset(captions_file, video_dir, output_dir, num_spliced_dataset_bins=10, text_model="openai/clip-vit-base-patch32", vae_model="hpcai-tech/vqvae"):
    tokenizer = AutoTokenizer.from_pretrained(text_model)
    text_model = CLIPTextModel.from_pretrained(text_model).cuda().eval()
    vqvae = AutoModel.from_pretrained(vae_model, trust_remote_code=True).cuda().eval()

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
    
    ds = load_dataset("json", data_files=captions_file, keep_in_memory=False, split=train_splits)

    for i, part_ds in enumerate(ds):
        print(f"Processing part {i+1}/{len(ds)}")
        part_ds = part_ds.map(process_batch, 
                              fn_kwargs={
                                  "video_dir": video_dir, 
                                  "tokenizer": tokenizer, 
                                  "text_model": text_model, 
                                  "vqvae": vqvae
                                  }, 
                                  batched=True, 
                                  batch_size=1,
                                  keep_in_memory=False,
            remove_columns=part_ds.column_names)
        output_path = os.path.join(output_dir, f"part-{i:05d}")
        part_ds.save_to_disk(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument("captions_file", type=str, help="Path to the captions file. It should be a JSON file or a JSONL file")
    parser.add_argument("video_dir", type=str, help="Path to the video directory")
    parser.add_argument("output_dir", type=str, help="Path to the output directory")
    parser.add_argument("-n", "--num_spliced_dataset_bins", type=int, default=10, help="Number of bins for spliced dataset")
    parser.add_argument("--text_model", type=str, default="openai/clip-vit-base-patch32", help="CLIP text model")
    parser.add_argument("--vae_model", type=str, default="hpcai-tech/vqvae", help="VQ-VAE model")
    args = parser.parse_args()
    process_dataset(args.captions_file, args.video_dir, args.output_dir, args.num_spliced_dataset_bins, args.text_model, args.vae_model)

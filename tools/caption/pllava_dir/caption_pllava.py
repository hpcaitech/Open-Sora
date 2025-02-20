import os
import sys
from pathlib import Path

current_file = Path(__file__)  # Gets the path of the current file
fourth_level_parent = current_file.parents[3]

datasets_dir = os.path.join(fourth_level_parent, "opensora/datasets")
import sys

sys.path.append(datasets_dir)
from read_video import read_video_av

sys.path.remove(datasets_dir)

import itertools
import logging
import random
import traceback
from argparse import ArgumentParser
from multiprocessing import Process, Queue

import colossalai
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torchvision
import transformers
from colossalai.utils import get_current_device
from PIL import Image
from tasks.eval.eval_utils import Conversation
from tasks.eval.model_utils import load_pllava
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--prompt_template", type=str, default="general", choices=["general", "person"])
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        required=True,
        default=4,
    )
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument(
        "--lora_alpha",
        type=int,
        required=False,
        default=4,
    )
    parser.add_argument(
        "--weight_dir",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--conv_mode",
        type=str,
        required=False,
        default="eval_mvbench",
    )
    parser.add_argument(
        "--pooling_shape",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--error_message",
        type=str,
        required=False,
        default="error occured during captioning",
    )
    parser.add_argument("--keep_failed", action="store_true", default=False)
    parser.add_argument(
        "--short_caption_ratio",
        type=float,
        required=False,
        default=0,
    )

    args = parser.parse_args()
    return args


###############
# data processing
###############


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([start + int(np.round(seg_size * idx)) for idx in range(num_segments)])
    return offsets


def load_video(video_path, num_frames, return_msg=False, resolution=336):
    transforms = torchvision.transforms.Resize(size=resolution)
    # vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    vframes, aframes, info = read_video_av(video_path, pts_unit="sec", output_format="THWC")
    # print(vframes.shape)
    total_num_frames = len(vframes)
    # print("Video path: ", video_path)
    # print("Total number of frames: ", total_num_frames)
    frame_indices = get_index(total_num_frames, num_frames)
    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vframes[frame_index].numpy())
        images_group.append(transforms(img))
    if return_msg:
        # fps = float(vframes.get_avg_fps())
        # sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # # " " should be added in the start and end
        # msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        # return images_group, msg
        exit("return_msg not implemented yet")
    else:
        return images_group


class CSVDataset(Dataset):
    def __init__(self, csv_path, num_frames):
        self.df = pd.read_csv(csv_path)
        self.data_list = self.df.path.tolist()
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        try:
            video = load_video(self.data_list[idx], self.num_frames, resolution=RESOLUTION)
            return video
        except:
            return None

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None, None

        if random.random() <= SHORT_CAPTION_RATIO:
            prompt = SHORT_PROMPT
            max_tokens = MAX_SHORT_TOKENS
        else:
            prompt = LONG_PROMPT
            max_tokens = MAX_LONG_TOKENS

        processed_batch = [processor(text=prompt, images=video, return_tensors="pt") for video in batch]
        batch = default_collate(processed_batch)

        for k, v in batch.items():
            if k in ("input_ids", "attention_mask"):
                batch[k] = v.squeeze(1)
            elif k == "pixel_values":
                b, t, c, h, w = v.shape
                batch[k] = v.reshape(b * t, c, h, w)
        return batch, max_tokens

    @staticmethod
    def post_process(output_texts, processor):
        output_texts = processor.batch_decode(
            output_texts, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if LONG_CONV_TEMPLATE.roles[-1] == "<|im_start|>assistant\n":
            split_tag = "<|im_start|> assistant\n"
        else:
            split_tag = LONG_CONV_TEMPLATE.roles[-1]
        ending = LONG_CONV_TEMPLATE.sep if isinstance(LONG_CONV_TEMPLATE.sep, str) else LONG_CONV_TEMPLATE.sep[1]
        for i, output_text in enumerate(output_texts):
            output_text = output_text.split(split_tag)[-1]
            output_text = output_text.removesuffix(ending).strip()
            output_text = output_text.replace("\n", " ")
            output_texts[i] = output_text
        return output_texts


def load_model_and_dataset(
    pretrained_model_name_or_path,
    num_frames,
    use_lora,
    lora_alpha,
    weight_dir,
    csv_path,
    pooling_shape=(16, 12, 12),
):
    # remind that, once the model goes larger (30B+) may cause the memory to be heavily used up. Even Tearing Nodes.
    model, processor = load_pllava(
        pretrained_model_name_or_path,
        num_frames=num_frames,
        use_lora=use_lora,
        weight_dir=weight_dir,
        lora_alpha=lora_alpha,
        pooling_shape=pooling_shape,
    )

    #  position embedding
    model = model.to(device=get_current_device())
    model = model.eval()

    dataset = CSVDataset(csv_path, num_frames)
    return model, processor, dataset


def infer(
    model,
    batch,
    max_tokens,
):
    batch = batch.to(get_current_device())
    with torch.no_grad():
        output_texts = model.generate(
            **batch,
            media_type="video",
            do_sample=False,
            max_new_tokens=max_tokens,
            num_beams=1,
            min_length=1,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=1.0,
        )
    output_texts = [x.cpu() for x in output_texts]
    return output_texts


def inference_loop(args, model, dataset, q: Queue):
    dataloader = DataLoader(
        dataset,
        num_workers=2,
        batch_size=args.batch_size,
        collate_fn=CSVDataset.collate_fn,
        pin_memory=True,
        sampler=DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False),
    )

    for i, (batch, max_tokens) in enumerate(tqdm(dataloader, disable=dist.get_rank() != 0)):
        try:
            if batch is None:
                raise Exception("Video not loaded properly")
            preds = infer(
                model,
                batch,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"error at rank {dist.get_rank()} sample {i}: {str(e)}")
            traceback.print_exception(e)
            # preds = args.error_message duplicated for each video in the batch
            preds = [args.error_message] * len(batch)
        q.put(preds)
    # finish the queue
    q.put(None)


def post_process_loop(processor, q: Queue, result_q: Queue):
    results = []
    while True:
        preds = q.get()
        if preds is not None:
            preds = CSVDataset.post_process(preds, processor)
            results.extend(preds)
        else:
            break
    result_q.put(results)


def main():
    args = parse_args()
    if args.prompt_template == "general":
        long_pt = "Describe this video. Pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. Here are some examples of good descriptions: 1. A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about. 2. Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field. 3. Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff's edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff’s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway."
        short_pt = "Describe the video focusing on key objects and actions. The description should be brief yet detailed enough for AI to recreate the video. Keep the description to no more than three sentences. Here are some examples of good descriptions: 1. A stylish woman walks confidently down a neon-lit Tokyo street, wearing a black leather jacket and a long red dress, with pedestrians and reflective wet pavement around her. 2. Giant wooly mammoths tread through a snowy meadow, their fur blowing lightly in the wind, with snowy trees and mountains in the background. 3. A drone captures waves crashing against rugged cliffs along Big Sur, with golden sunset light illuminating the rocky shore and a lighthouse in the distance."
    elif args.prompt_template == "person":
        # pt = "Describe this video in detail. Pay special attention to all details of the person, including the face, the body, the pose, the action, and the outfit. Also pay attention to the camera angle. The description should be useful for AI to re-generate the video. The description should contain no more than six sentences."
        long_pt = "Describe this video in detail. Pay special attention to all details of the person, including 1. apperance, such as hair, face, body, and outfit; 2. expression and emotion; 3. action and pose. Also pay attention to the background and the surrounding environment. Also pay attention to the camera angle. The description should be useful for AI to re-generate the video. The description should contain no more than six sentences."
        short_pt = "Describe this video in detail. Pay special attention to key details of the person, including 1. apperance, such as hair, face, body, and outfit; 2. expression and emotion; 3. action and pose. Also pay attention to the background and the surrounding environment. Also pay attention to the camera angle. The description should be useful for AI to re-generate the video. The description should contain no more than three sentences."
    else:
        raise ValueError

    assert (
        args.short_caption_ratio >= 0 and args.short_caption_ratio <= 1
    ), "`short_caption_ratio` should be in range [0, 1]"

    global LONG_CONV_TEMPLATE
    global SHORT_CONV_TEMPLATE
    global LONG_PROMPT
    global SHORT_PROMPT
    global RESOLUTION
    global SHORT_CAPTION_RATIO
    global MAX_LONG_TOKENS
    global MAX_SHORT_TOKENS

    LONG_CONV_TEMPLATE = Conversation(
        system=long_pt,
        roles=("USER:", "ASSISTANT:"),
        messages=[],
        sep=(" ", "</s>"),
        mm_token="<image>",
    )
    LONG_CONV_TEMPLATE.user_query("Describe the video in detail.", is_mm=True)
    LONG_PROMPT = LONG_CONV_TEMPLATE.get_prompt()

    SHORT_CONV_TEMPLATE = Conversation(
        system=short_pt,
        roles=("USER:", "ASSISTANT:"),
        messages=[],
        sep=(" ", "</s>"),
        mm_token="<image>",
    )
    SHORT_CONV_TEMPLATE.user_query("Describe the video in detail.", is_mm=True)
    SHORT_PROMPT = SHORT_CONV_TEMPLATE.get_prompt()

    RESOLUTION = 672
    SHORT_CAPTION_RATIO = args.short_caption_ratio
    MAX_LONG_TOKENS = 256
    MAX_SHORT_TOKENS = 128

    colossalai.launch_from_torch()
    rank = dist.get_rank()

    # setup debug
    if rank == 0:
        import os

        if os.getenv("DEBUG_ADDRESS") != None:
            import ptvsd

            ptvsd.enable_attach(address=("localhost", int(os.getenv("DEBUG_ADDRESS"))), redirect_output=True)
            ptvsd.wait_for_attach()
            print("waiting for debugger attachment")
    else:
        transformers.utils.logging.set_verbosity_error()
        logger.setLevel(transformers.logging.ERROR)

    # setup model and dataset
    if args.pooling_shape is not None:
        pooling_shape = tuple([int(x) for x in args.pooling_shape.split("-")])

    global processor
    model, processor, dataset = load_model_and_dataset(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        num_frames=args.num_frames,
        use_lora=args.use_lora,
        lora_alpha=args.lora_alpha,
        weight_dir=args.weight_dir,
        pooling_shape=pooling_shape,
        csv_path=args.csv_path,
    )
    logger.info(f"Dataset loaded with {len(dataset)} samples.")
    q = Queue()
    result_q = Queue()
    p = Process(target=post_process_loop, args=(processor, q, result_q))
    p.start()

    inference_loop(args, model, dataset, q)
    results = result_q.get()
    p.join()

    # gather results
    results_list = [None for _ in range(dist.get_world_size())] if rank == 0 else None
    dist.gather_object(results, results_list, dst=0)
    if rank == 0:
        # reorder and merge
        final_results = list(itertools.chain.from_iterable(zip(*results_list)))
        assert len(final_results) >= len(dataset)
        # remove padding
        final_results = final_results[: len(dataset)]

        # write the results to the csv file
        df = pd.read_csv(args.csv_path)
        # add a new column to the dataframe
        df["text"] = final_results
        drop_failed = not args.keep_failed
        if drop_failed:
            df = df[df["text"] != args.error_message]
            print(f"Dropped {len(dataset) - len(df)} samples")
        # write the dataframe to a new csv file called '*_pllava_13b_caption.csv'
        new_csv_path = args.csv_path.replace(".csv", "_text.csv")
        df.to_csv(new_csv_path, index=False)
        print(f"Results saved to {new_csv_path}")


if __name__ == "__main__":
    main()

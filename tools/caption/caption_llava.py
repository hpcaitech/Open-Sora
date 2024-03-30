import argparse
import csv
import os
import time
import warnings
from datetime import timedelta

import pandas as pd
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator, ProcessGroupMesh
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.utils import get_current_device, set_seed
from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_anyres_image_grid_shape, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.llava_arch import unpad_image
from llava.utils import disable_torch_init
from PIL import Image
from tqdm import tqdm

from .acceleration.llava.policy import LlavaForCausalLMPolicy
from .utils import Timer, extract_frames, prompts

disable_torch_init()

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def is_video(filename):
    ext = os.path.splitext(filename)[-1].lower()
    return ext in VID_EXTENSIONS


def get_image(image_path):
    return Image.open(image_path).convert("RGB")


def prepare_inputs_labels_for_multimodal(
    self, input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=None
):
    # llava_arch.py
    vision_tower = self.get_vision_tower()
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    if type(images) is list or images.ndim == 5:
        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
        concat_images = torch.cat([image for image in images], dim=0)
        image_features = self.encode_images(concat_images)
        split_sizes = [image.shape[0] for image in images]
        image_features = torch.split(image_features, split_sizes, dim=0)
        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
        if mm_patch_merge_type == "flat":
            image_features = [x.flatten(0, 1) for x in image_features]
        elif mm_patch_merge_type.startswith("spatial"):
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.get_vision_tower().num_patches_per_side
                    assert height * width == base_image_feature.shape[0]
                    if image_aspect_ratio == "anyres":
                        num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                            image_sizes[image_idx],
                            self.config.image_grid_pinpoints,
                            self.get_vision_tower().config.image_size,
                        )
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    else:
                        raise NotImplementedError
                    if "unpad" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        image_feature = torch.cat(
                            (
                                image_feature,
                                self.model.image_newline[:, None, None]
                                .expand(*image_feature.shape[:-1], 1)
                                .to(image_feature.device),
                            ),
                            dim=-1,
                        )
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        image_feature = image_feature.flatten(0, 3)
                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                else:
                    image_feature = image_feature[0]
                    if "unpad" in mm_patch_merge_type:
                        image_feature = torch.cat(
                            (image_feature, self.model.image_newline[None].to(image_feature.device)), dim=0
                        )
                new_image_features.append(image_feature)
            image_features = new_image_features
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    else:
        image_features = self.encode_images(images)

    # TODO: image start / end is not implemented here to support pretraining.
    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
        raise NotImplementedError

    # Let's just add dummy tensors if they do not exist,
    # it is a headache to deal with None all the time.
    # But it is not ideal, and if you have a better idea,
    # please open an issue / submit a PR, thanks.
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- FIXME
    input_ids = [
        cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
    ]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = (
            [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        )
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(
                    torch.full(
                        (cur_image_features.shape[0],),
                        IGNORE_INDEX,
                        device=cur_labels.device,
                        dtype=cur_labels.dtype,
                    )
                )

        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    # Truncate sequences to max length as image embeddings can make the sequence longer
    tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
    if tokenizer_model_max_length is not None:
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    # Combine them
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full(
        (batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device
    )
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, "tokenizer_padding_side", "right") == "left":
            new_input_embeds_padded.append(
                torch.cat(
                    (
                        torch.zeros(
                            (max_len - cur_len, cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                        cur_new_embed,
                    ),
                    dim=0,
                )
            )
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(
                    0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                )
        else:
            new_input_embeds_padded.append(
                torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_len, cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
            )
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(
                    0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                )

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None, points=(0.1, 0.5, 0.9)):
        self.csv_path = csv_path
        self.transform = transform
        self.data = pd.read_csv(csv_path)
        self.points = points

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = sample["path"]
        if not is_video(path):
            images = [get_image(path)]
            length = 1
        else:
            images, length = extract_frames(sample["path"], points=self.points)
        imgs_size = [img.size for img in images]
        images = self.transform(images)

        # we put images into a list as pytorch dataloader does not accept Pill
        out = dict(path=path, image=images, length=length, img_size=imgs_size)
        return out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.getitem(index)


def collate_fn(batch):
    paths = [item["path"] for item in batch]
    images = [item["image"] for item in batch]
    lengths = [item["length"] for item in batch]
    img_sizes = [item["img_size"] for item in batch]
    return paths, images, lengths, img_sizes


@torch.inference_mode()
def main(args):
    # ======================================================
    # 1. init environment
    # ======================================================
    # we set a very large timeout to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(1024)
    coordinator = DistCoordinator()

    # prepare the dp and tp groups
    assert (
        args.dp_size * args.tp_size == coordinator.world_size
    ), f"DP size {args.dp_size} * TP size {args.tp_size} must equal to world size {coordinator.world_size}"
    mesh = ProcessGroupMesh(args.dp_size, args.tp_size)
    dp_group = mesh.get_group_along_axis(0)
    tp_group = mesh.get_group_along_axis(1)

    # ======================================================
    # 3. load model and prepare prompts
    # ======================================================
    model_path = "liuhaotian/llava-v1.6-34b"
    query = prompts[args.prompt]
    print(f"Prompt: {query}")
    conv = conv_templates["chatml_direct"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + query)
    prompt = conv.get_prompt()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Pytorch non-meta copying warning fills out the console
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device=get_current_device(),
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
        dist.barrier()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    # ======================================================
    # 4. Apply system optimization
    # ======================================================
    # create huggingface model as normal
    tp_size = dist.get_world_size(tp_group)
    shard_config = ShardConfig(
        tensor_parallel_process_group=tp_group if tp_size > 1 else None,
        enable_tensor_parallelism=True if tp_size > 1 else False,
    )
    shard_former = ShardFormer(shard_config=shard_config)
    model = shard_former.optimize(model, policy=LlavaForCausalLMPolicy())[0].cuda()
    torch.cuda.empty_cache()

    # ======================================================
    # 5. Prepare dataloader
    # ======================================================
    # build dataset
    def transform(imgs):
        imgs = process_images(imgs, image_processor, model.config)
        imgs = imgs.to(dtype=torch.float16)
        return imgs

    dataset = VideoTextDataset(args.input, points=(0.2, 0.5, 0.8), transform=transform)
    total_num_videos = len(dataset)

    # build sampler
    dp_rank = dist.get_rank(dp_group)
    dp_size = dist.get_world_size(dp_group)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, rank=dp_rank, num_replicas=dp_size, shuffle=False
    )

    # build dataloader

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    # prepare output file reader
    output_file = args.input.replace(".csv", "_caption.csv")

    # create csv writer
    has_main_writer = dist.get_rank() == 0
    has_dp_writter = dist.get_rank(tp_group) == 0

    if has_main_writer:
        # we keep track of the processed videos in main file
        # so we use append mode
        main_file = open(output_file, "a")
        main_writer = csv.writer(main_file)

    if has_dp_writter:
        # the dp writer takes care of the files processed on the current dp rank
        # so we use write mode
        output_file_split = f"{output_file}.part{dp_rank}"
        dp_file = open(output_file_split, "w")
        dp_writer = csv.writer(dp_file)

    # ======================================================
    # 5. generate captions
    # ======================================================
    args.bs

    if dist.get_rank(tp_group) == 0:
        pbar = tqdm(dataloader, position=dp_rank, desc=f"Data Parallel Rank {dist.get_rank(dp_group)}")
    else:
        pbar = dataloader

    if args.profile:
        encode_time = []
        frame_extraction_time = []
        generate_time = []
        total_time = []

    for batch in pbar:
        # measure time
        if args.profile:
            torch.cuda.synchronize()
            start_time = time.time()

        video_files, frames, video_lengths, img_size_list = batch

        # encode the batch of inputs
        with Timer() as encode_timer:
            samples = []
            for imgs, imgs_size in zip(frames, img_size_list):
                imgs = imgs.cuda()
                _, _, _, _, inputs_embeds, _ = prepare_inputs_labels_for_multimodal(
                    model, input_ids, None, None, None, None, images=imgs, image_sizes=imgs_size
                )
                samples.append(inputs_embeds)

        # padding
        max_len = max([sample.shape[1] for sample in samples])
        attention_mask = torch.tensor(
            [[0] * (max_len - samples[i].shape[1]) + [1] * samples[i].shape[1] for i in range(len(samples))]
        ).to(model.device)
        inputs_embeds = [
            torch.cat(
                [
                    torch.zeros(
                        (1, max_len - samples[i].shape[1], samples[i].shape[-1]),
                        device=model.device,
                        dtype=torch.float16,
                    ),
                    samples[i],
                ],
                dim=1,
            )
            for i in range(len(samples))
        ]
        inputs_embeds = torch.cat(inputs_embeds, dim=0)

        # generate outputs
        with Timer() as generate_timer:
            output_ids = super(type(model), model).generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=args.tp_size == 1,  # sampling is not deterministic and may cause TP to hang
                temperature=0.2,
                max_new_tokens=512,
                use_cache=True,
            )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            outputs = [output.replace("\n", " ").strip() for output in outputs]

        # warmup for 1 iter
        if args.profile and i < args.profile_warmup:
            # measure time
            torch.cuda.synchronize()
            time_taken = time.time() - start_time

            total_time.append(time_taken)
            encode_time.append(encode_timer.time_taken)
            generate_time.append(generate_timer.time_taken)

        # save results
        if has_dp_writter:
            result = list(zip(video_files, outputs, video_lengths))
            for t in result:
                dp_writer.writerow(t)

    # display profiling info
    if args.profile:
        num_samples_after_warmup = total_num_videos - args.bs * args.profile_warmup * dp_size
        print(f"throughput (video/s): {num_samples_after_warmup / sum(total_time)}")
        print(f"average frame extraction time per sample: {sum(frame_extraction_time) / num_samples_after_warmup}")
        print(f"average encode time per sample: {sum(encode_time) / num_samples_after_warmup}")
        print(f"average generate time per sample: {sum(generate_time) / num_samples_after_warmup}")
        print(f"Max GPU allocated / GB: {torch.cuda.max_memory_allocated() / 1024**3}")
        print(f"Max GPU reserved / GB: {torch.cuda.max_memory_reserved() / 1024**3}")

    # ======================================================
    # 6. shutdown
    # ======================================================
    # close file writing
    if has_dp_writter:
        dp_file.close()
    dist.barrier()

    # merge files
    if has_main_writer:
        for i in range(dp_size):
            output_file_split = f"{output_file}.part{i}"
            with open(output_file_split, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    main_writer.writerow(row)
            os.remove(output_file_split)
        main_file.close()

    # terminate distributed env
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to the input CSV file")

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--prompt", type=str, default="three_frames")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--prefetch-factor", type=int, default=8, help="Prefetch factor")
    args = parser.parse_args()
    main(args)

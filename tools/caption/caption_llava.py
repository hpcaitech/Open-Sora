import argparse
import csv
import os
import warnings

import torch
from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_anyres_image_grid_shape, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.llava_arch import unpad_image
from llava.utils import disable_torch_init
from tqdm import tqdm

from .utils import extract_frames, prompts, read_video_list

disable_torch_init()


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


@torch.inference_mode()
def main(args):
    # ======================================================
    # 1. read video list
    # ======================================================
    videos = read_video_list(args.video_folder, args.output_file)
    f = open(args.output_file, "a")
    writer = csv.writer(f)

    # ======================================================
    # 2. load model and prepare prompts
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
        )
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(model.device)

    # ======================================================
    # 3. generate captions
    # ======================================================
    bs = args.bs
    for i in tqdm(range(0, len(videos), bs)):
        # prepare a batch of inputs
        video_files = videos[i : i + bs]
        frames = []
        video_lengths = []
        for video_file in video_files:
            frame, length = extract_frames(os.path.join(args.video_folder, video_file))
            if len(frame) < 3:
                continue
            frames.append(frame)
            video_lengths.append(length)
        if len(frames) == 0:
            continue

        # encode the batch of inputs
        samples = []
        for imgs in frames:
            imgs_size = [img.size for img in imgs]
            imgs = process_images(imgs, image_processor, model.config)
            imgs = imgs.to(model.device, dtype=torch.float16)
            with torch.inference_mode():
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
        output_ids = super(type(model), model).generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=512,
            use_cache=True,
        )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        outputs = [output.replace("\n", " ").strip() for output in outputs]

        # save results
        result = list(zip(video_files, outputs, video_lengths))
        for t in result:
            writer.writerow(t)

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_folder", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--prompt", type=str, default="three_frames")
    args = parser.parse_args()

    main(args)

import argparse
import csv
import time
import warnings
from datetime import timedelta

import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator, ProcessGroupMesh
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.utils import get_current_device, set_seed
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ..datasets.utils import IMG_EXTENSIONS, VID_EXTENSIONS
from .acceleration.llava.policies import LlavaLlamaForCausalLMPolicy, LlavaMistralForCausalLMPolicy
from .utils import PROMPTS, Timer, VideoTextDataset, collate_fn

disable_torch_init()


class NoPaddingDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, seed=seed, shuffle=False, drop_last=False
        )
        remainder = len(self.dataset) % self.num_replicas
        if remainder > 0 and (self.rank + 1) - remainder <= 0:
            # if the dataset is not divisible by num_replicas
            # the remaining items will be allocated to the first n ranks
            self.num_samples = len(self.dataset) // self.num_replicas + 1
        else:
            self.num_samples = len(self.dataset) // self.num_replicas
        self.total_size = len(dataset)

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # remove tail of data to make it evenly divisible.
        indices = indices[: self.total_size]

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)


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
    # 2. load model
    # ======================================================
    model_path = args.model_path
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Pytorch non-meta copying warning fills out the console
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device=get_current_device(),
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2" if args.flash_attention else "eager",
        )
        dist.barrier()

    # ======================================================
    # 3. Apply system optimization
    # ======================================================
    tp_size = dist.get_world_size(tp_group)
    shard_config = ShardConfig(
        tensor_parallel_process_group=tp_group if tp_size > 1 else None,
        enable_tensor_parallelism=True if tp_size > 1 else False,
    )
    shard_former = ShardFormer(shard_config=shard_config)

    # check the model type
    model_name = model.__class__.__name__
    print(model_name)
    if model_name == "LlavaLlamaForCausalLM":
        model = shard_former.optimize(model, policy=LlavaLlamaForCausalLMPolicy())[0].cuda()
    elif model_name == "LlavaMistralForCausalLM":
        model = shard_former.optimize(model, policy=LlavaMistralForCausalLMPolicy())[0].cuda()
    else:
        print(f"The shardformer policy for {model_name} is not implemented, skip")
    torch.cuda.empty_cache()

    # ======================================================
    # 4. Prepare dataloader
    # ======================================================
    # prepare prompt
    query = PROMPTS[args.prompt]["text"]
    if dist.get_rank() == 0:
        print(f"Prompt: {query}")

    if "text" in args.prompt:

        def get_text_input_ids(text):
            conv = conv_templates["chatml_direct"].copy()
            query_text = query.format(text)
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + query_text)
            prompt = conv.get_prompt()
            # add num_frames images
            t = prompt.split("<image>")
            prompt = t[0] + "<image>" * args.num_frames + t[1]
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0)
            return input_ids

    else:
        conv = conv_templates["chatml_direct"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + query)
        prompt = conv.get_prompt()
        # add num_frames images
        t = prompt.split("<image>")
        prompt = t[0] + "<image>" * args.num_frames + t[1]
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0)

        def get_text_input_ids(*args):
            return input_ids

    # build dataset
    def transform(imgs):
        imgs = process_images(imgs, image_processor, model.config)
        imgs = imgs.to(dtype=torch.float16)
        return imgs

    dataset = VideoTextDataset(
        args.input,
        transform=transform,
        num_frames=args.num_frames,
        get_text_input_ids=get_text_input_ids,
        resize=args.resize,
    )

    # make sure that the prompt type matches the data type
    data_extension = "." + dataset.data["path"].iloc[0].split(".")[-1]
    prompt_type = PROMPTS[args.prompt]["type"]
    if prompt_type == "image":
        assert (
            data_extension.lower() in IMG_EXTENSIONS
        ), f"The prompt is suitable for an image dataset but the data is not image. The first data is of format {data_extension}"
    elif prompt_type == "video":
        assert (
            data_extension.lower() in VID_EXTENSIONS
        ), f"The prompt is suitable for a video dataset but the data is not video. The first data is of format {data_extension}"
    else:
        raise ValueError(f"Found invalid prompt type {prompt_type}")

    total_num_videos = len(dataset)

    # build sampler
    dp_rank = dist.get_rank(dp_group)
    dp_size = dist.get_world_size(dp_group)
    sampler = NoPaddingDistributedSampler(dataset, rank=dp_rank, num_replicas=dp_size)

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
    has_dp_writter = dist.get_rank(tp_group) == 0

    if has_dp_writter:
        # the dp writer takes care of the files processed on the current dp rank
        # so we use write mode
        output_file_split = output_file.replace(".csv", f"_part{dp_rank}.csv")
        dp_file = open(output_file_split, "w")
        dp_writer = csv.writer(dp_file)
        dp_writer.writerow(["path", "text", "num_frames"])

    # ======================================================
    # 5. generate captions
    # ======================================================
    if dist.get_rank(tp_group) == 0:
        pbar = tqdm(dataloader, position=dp_rank, desc=f"Data Parallel Rank {dist.get_rank(dp_group)}")
    else:
        pbar = dataloader

    if args.profile:
        encode_time = []
        generate_time = []
        output_length = []
        total_time = []

    for i, batch in enumerate(pbar):
        # measure time
        if args.profile:
            torch.cuda.synchronize()
            start_time = time.time()

        video_files, frames, video_lengths, img_size_list, texts = batch

        # encode the batch of inputs
        with Timer() as encode_timer:
            samples = []
            for imgs, imgs_size, input_ids in zip(frames, img_size_list, texts):
                imgs = imgs.cuda()
                input_ids = input_ids.cuda()
                _, _, _, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, None, None, None, images=imgs, image_sizes=imgs_size
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
                do_sample=False,  # sampling is not deterministic and may cause TP to hang
                max_new_tokens=args.max_tokens,
                use_cache=True,
            )

            # skip warmup and add profiling data
            if args.profile and i >= args.profile_warmup:
                output_length.append(output_ids.size(0) * output_ids.size(1))

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            outputs = [output.replace("\n", " ").strip() for output in outputs]

        # skip warmup and add profiling data
        if args.profile and i >= args.profile_warmup:
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
        print(output_length)
        num_samples_after_warmup = total_num_videos - args.bs * args.profile_warmup * dp_size
        print(f"throughput (samples/s): {num_samples_after_warmup / sum(total_time)}")
        print(f"average encode time per sample: {sum(encode_time) / num_samples_after_warmup}")
        print(f"average generate time per sample: {sum(generate_time) / num_samples_after_warmup}")
        print(f"average number of tokens characters per sample: {sum(output_length) / num_samples_after_warmup}")
        print(f"Max GPU allocated / GB: {torch.cuda.max_memory_allocated() / 1024**3}")
        print(f"Max GPU reserved / GB: {torch.cuda.max_memory_reserved() / 1024**3}")

    # ======================================================
    # 6. shutdown
    # ======================================================
    # close file writing
    if has_dp_writter:
        dp_file.close()
    dist.barrier()

    # terminate distributed env
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to the input CSV file")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.6-34b")
    parser.add_argument("--prompt", type=str, default="video-f1-detail-3ex")
    parser.add_argument("--resize", type=int, default=336)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=300)
    # speed related
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--dp-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=8, help="Prefetch factor")
    parser.add_argument(
        "--flash-attention",
        action="store_true",
        help="Whether to use flash attention. You can turn on this flag for llama model and off for mistral model.",
    )
    # debug related
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-warmup", type=int, default=1)

    args = parser.parse_args()
    main(args)

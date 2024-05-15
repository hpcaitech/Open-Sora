import pandas as pd
import functools
import itertools
import logging
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
import multiprocessing as mp
from argparse import ArgumentParser
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision
from decord import VideoReader, cpu
import transformers
import pandas as pd
# import sys
# sys.path.append('/home/tom/Open-Sora-dev/tools/caption/pllava_dir/PLLaVA')
from tasks.eval.model_utils import load_pllava, pllava_answer
from tasks.eval.eval_utils import Conversation


conv_template = Conversation(
    system="Describe this video. Pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. Here are some examples of good descriptions: 1. A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about. 2. Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field. 3. Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff's edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff’s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.",
    roles=("USER:", "ASSISTANT:"),
    messages=[],
    sep=(" ", "</s>"),
    mm_token='<image>'
)


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RESOLUTION = 672 # 

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def load_video(video_path, num_frames, return_msg=False, resolution=336):
    transforms = torchvision.transforms.Resize(size=resolution)
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_num_frames = len(vr)
    frame_indices = get_index(total_num_frames, num_frames)
    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        images_group.append(transforms(img))
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return images_group, msg
    else:
        return images_group

class CSVDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.data_list = self.df.path.tolist()
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data_list):
            raise IndexError
        # print(len(self.data_list))
        return self.data_list[idx]

    def set_rank_and_world_size(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.data_per_gpu = len(self) // world_size
        start_index = rank * self.data_per_gpu
        end_index = (rank + 1) * self.data_per_gpu if rank != world_size - 1 else len(self)
        self.data_list = self.data_list[start_index:end_index]

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        default='llava-hf/llava-1.5-7b-hf'
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
    parser.add_argument(
        "--use_lora",
        action='store_true'
    )
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
        default='eval_mvbench',
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
        default=None,
    )
    args = parser.parse_args()
    return args

def load_model_and_dataset(rank, world_size, pretrained_model_name_or_path, num_frames, use_lora, lora_alpha, weight_dir, csv_path, pooling_shape=(16,12,12)):
    # remind that, once the model goes larger (30B+) may cause the memory to be heavily used up. Even Tearing Nodes.
    model, processor = load_pllava(pretrained_model_name_or_path, num_frames=num_frames, use_lora=use_lora, weight_dir=weight_dir, lora_alpha=lora_alpha, pooling_shape=pooling_shape)
    logger.info('done loading llava')

    #  position embedding
    model = model.to(torch.device(rank))
    model = model.eval()

    dataset = CSVDataset(csv_path)
    dataset.set_rank_and_world_size(rank, world_size)
    return model, processor, dataset

def infer(\
        model,
        processor,
        path,
        conv_mode,
        num_frames,
        print_res=True,
    ):
    video_list = load_video(path, num_frames, resolution=RESOLUTION)
    conv = conv_template.copy()
    conv.user_query("Describe the video in details.", is_mm=True)

    llm_response, conv = pllava_answer(
        conv=conv,
        model=model,
        processor=processor,
        img_list=video_list,
        max_new_tokens=256,
        do_sample=False,
        print_res=print_res
    )
    
    return llm_response

def run(rank, args, world_size):
    if rank == 0:
        # import ptvsd; ptvsd.enable_attach(address=('localhost', 1025), redirect_output=True); ptvsd.wait_for_attach(); print('waiting for debugger attachment')
        pass
    if rank != 0:
        transformers.utils.logging.set_verbosity_error()
        logger.setLevel(transformers.logging.ERROR)

    print_res = False
    conv_mode= args.conv_mode
    if args.pooling_shape is not None:
        pooling_shape=tuple([int(x) for x in args.pooling_shape.split("-")])

    logger.info(f'loading model and constructing dataset to gpu {rank}...')
    model, processor, dataset = load_model_and_dataset(rank,
                                                       world_size,
                                                       pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                                       num_frames=args.num_frames,
                                                       use_lora=args.use_lora,
                                                       lora_alpha=args.lora_alpha,
                                                       weight_dir=args.weight_dir,
                                                       pooling_shape=pooling_shape,
                                                       csv_path=args.csv_path)
    logger.info(f'done model and dataset...')
    logger.info('constructing dataset...')
    logger.info('single test...')
    
    tbar = tqdm(total=len(dataset))
    total = 0
    result_list = []
    done_count = 0
    print(len(dataset))
    for example in dataset:
        total += 1
        try:
            pred = infer(
                model,
                processor,
                example,
                conv_mode=conv_mode,
                num_frames=args.num_frames,
                print_res=print_res,
            )
        except Exception as e:
            logger.error(f'error in {example}: {str(e)}')
            pred = args.error_message
        result_list.append(pred)
        tbar.update(len(result_list) - done_count)
        done_count = len(result_list)
    return result_list

def main():
    multiprocess = True
    mp.set_start_method('spawn')
    args = parse_args()
    # csv_path = '/home/tom/PLLaVA/test_short_caption_part2.csv'
    if multiprocess:
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        print(f'world_size: {world_size}')
        with Pool(world_size) as pool:
            func = functools.partial(run, args=args, world_size=world_size)
            result_lists = pool.map(func, range(world_size))
        
        logger.info('finished running')
        result_list = [res for res in itertools.chain(*result_lists)]
    else:
        result_list = run(0, world_size=1, args=args) # debug
    df = pd.read_csv(args.csv_path)
    # add a new column to the dataframe
    df['text'] = result_list 
    drop_failed = True
    if drop_failed:
        # iterate through the dataframe and delete the entire row if captioning failed
        for i in tqdm(range(len(df))):
            if df['text'][i] == args.error_message:
                df = df.drop(i)
    # write the dataframe to a new csv file called '*_pllava_13b_caption.csv'
    new_csv_path = args.csv_path.replace('.csv', '_text.csv')
    df.to_csv(new_csv_path, index=False)

if __name__ == "__main__":
    main()
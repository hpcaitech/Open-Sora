import numpy as np
import torch
from tqdm import tqdm
import math
from einops import rearrange
import sys
sys.path.append(".")
from flolpips.pwcnet import Network as PWCNet
from flolpips.flolpips import FloLPIPS

loss_fn = FloLPIPS(net='alex', version='0.1').eval().requires_grad_(False)
flownet = PWCNet().eval().requires_grad_(False)

def trans(x):
    return x


def calculate_flolpips(videos1, videos2, device):
    global loss_fn, flownet
    
    print("calculate_flowlpips...")
    loss_fn = loss_fn.to(device)
    flownet = flownet.to(device)
    
    if videos1.shape != videos2.shape:
        print("Warning: the shape of videos are not equal.")
        min_frames = min(videos1.shape[1], videos2.shape[1])
        videos1 = videos1[:, :min_frames]
        videos2 = videos2[:, :min_frames]
        
    videos1 = trans(videos1)
    videos2 = trans(videos2)

    flolpips_results = []
    for video_num in tqdm(range(videos1.shape[0])):
        video1 = videos1[video_num].to(device)
        video2 = videos2[video_num].to(device)
        frames_rec = video1[:-1]
        frames_rec_next = video1[1:]
        frames_gt = video2[:-1]
        frames_gt_next = video2[1:]
        t, c, h, w = frames_gt.shape
        flow_gt = flownet(frames_gt, frames_gt_next)
        flow_dis = flownet(frames_rec, frames_rec_next)
        flow_diff = flow_gt - flow_dis
        flolpips = loss_fn.forward(frames_gt, frames_rec, flow_diff, normalize=True)
        flolpips_results.append(flolpips.cpu().numpy().tolist())
        
    flolpips_results = np.array(flolpips_results) # [batch_size, num_frames]
    flolpips = {}
    flolpips_std = {}

    for clip_timestamp in range(flolpips_results.shape[1]):
        flolpips[clip_timestamp] = np.mean(flolpips_results[:,clip_timestamp], axis=-1)
        flolpips_std[clip_timestamp] = np.std(flolpips_results[:,clip_timestamp], axis=-1)

    result = {
        "value": flolpips,
        "value_std": flolpips_std,
        "video_setting": video1.shape,
        "video_setting_name": "time, channel, heigth, width",
        "result": flolpips_results,
        "details": flolpips_results.tolist()
    }

    return result

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 50
    CHANNEL = 3
    SIZE = 64
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)

    import json
    result = calculate_flolpips(videos1, videos2, "cuda:0")
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
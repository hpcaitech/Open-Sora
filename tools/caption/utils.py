import base64
import csv
import os

import cv2
from PIL import Image

prompts = {
    "naive": "Describe the video",
    "three_frames": "A video is given by providing three frames in chronological order. Describe this video and its style to generate a description. Pay attention to all objects in the video. Do not describe each frame individually. Do not reply with words like 'first frame'. The description should be useful for AI to re-generate the video. The description should be less than six sentences. Here are some examples of good descriptions: 1. A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about. 2. Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field. 3. Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff's edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff’s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.",
}


def get_filelist(file_path):
    Filelist = []
    VID_EXTENSIONS = ("mp4", "avi", "mov", "mkv")
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            ext = filename.split(".")[-1]
            if ext in VID_EXTENSIONS:
                Filelist.append(filename)
    return Filelist


def get_video_length(cap):
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_frames(video_path, points=(0.2, 0.5, 0.8), base_64=False):
    cap = cv2.VideoCapture(video_path)
    length = get_video_length(cap)
    points = [int(length * point) for point in points]
    frames = []
    if length < 3:
        return frames, length
    for point in points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, point)
        ret, frame = cap.read()
        if not base_64:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
        else:
            _, buffer = cv2.imencode(".jpg", frame)
            frame = base64.b64encode(buffer).decode("utf-8")
        frames.append(frame)
    return frames, length


def read_video_list(video_folder, output_file):
    processed_videos = []
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            reader = csv.reader(f)
            samples = list(reader)
            processed_videos = [sample[0] for sample in samples]

    # read video list
    videos = get_filelist(video_folder)
    print(f"Dataset contains {len(videos)} videos.")
    videos = [video for video in videos if video not in processed_videos]
    print(f"Processing {len(videos)} new videos.")
    return videos

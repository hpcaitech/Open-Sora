import argparse
import base64
import csv
import os

import cv2
import requests
import tqdm

# OpenAI API Key
api_key = ""


def get_caption(frame):
    prompt = "The middle frame from a video clip are given. Describe this video and its style to generate a description for the video. The description should be useful for AI to re-generate the video. Here are some examples of good descriptions:\n\n 1. A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.\n2. Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field.\n 3. Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff's edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff’s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway."

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}},
                ],
            }
        ],
        "max_tokens": 300,
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
    caption = response.json()["choices"][0]["message"]["content"]
    return caption


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    point = length // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, point)
    ret, frame = cap.read()
    _, buffer = cv2.imencode(".jpg", frame)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    return img_base64


def main(args):
    processed_videos = []
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            reader = csv.reader(f)
            samples = list(reader)
            processed_videos = [sample[0] for sample in samples]

    f = open(args.output_file, "a")
    writer = csv.writer(f)
    for video in tqdm.tqdm(os.listdir(args.video_folder)):
        if video in processed_videos:
            continue
        video_path = os.path.join(args.video_folder, video)
        base64_image = extract_frames(video_path)
        caption = get_caption(base64_image)
        caption = caption.replace("\n", " ")
        writer.writerow([video, caption])
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, required=True, help="Path to the folder containing the videos.")
    parser.add_argument("--output_file", type=str, default="video_captions.csv", help="Path to the output file.")
    args = parser.parse_args()
    main(args)

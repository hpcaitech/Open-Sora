_base_ = [
    "ucf-dit.py",
]

model = "PixArt-XL/2"
condition = "t5"
additional_model_args = dict(_delete_=True, space_scale=0.5)

ckpts = {
    "latest_2e-5": "outputs/093-F16S3-PixArt-XL-2/epoch7-global_step30000/",
    "latest_1e-4": "outputs/098-F16S3-PixArt-XL-2/epoch7-global_step30000/",
    "no_text": "outputs/107-F16S3-PixArt-XL-2/epoch3-global_step4000/",
    "text": "outputs/109-F16S3-PixArt-XL-2/epoch0-global_step1000/",
}
ckpt = ckpts["text"]
t5_path = "./pretrained_models/t5_ckpts"

labels = [
    "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.",  # sora
    "The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from it’s tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene.",  # sora
    "a couple of men hugging each other in a room",  # train
    "A man riding bike",  # ucf101
    "A woman in white walking in the desert",  # pika
]

# labels = [
#     "a picture of a person holding a knife in front of a refrigerator",
#     "a group of people sitting around a table",
#     "a woman smiling while sitting on a couch",
#     "Bright scene, aerial view,ancient city, fantasy, gorgeous light, mirror reflection, high detail, wide angle lens.",
# ]

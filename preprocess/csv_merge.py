import csv

# pexels_root = "/mnt/hdd/data/pexels/clips/fps-30_size-512_mins-1_maxs-8.3/"
# infos = (
#     (f"{pexels_root}animal", "/home/zhaowangbo/zangwei/minisora_new/pexels_animals_fnt.csv"),
#     (f"{pexels_root}art", "/home/zhaowangbo/zangwei/minisora_new/pexels_art.csv"),
#     (f"{pexels_root}balloon", "/home/zhaowangbo/zangwei/minisora_new/pexels_balloon_fnt.csv"),
#     (f"{pexels_root}car", "/home/zhaowangbo/zangwei/minisora_new/pexels_car_fnt.csv"),
#     (f"{pexels_root}city", "/home/zhaowangbo/zangwei/minisora_new/pexels_city_fnt.csv"),
#     (f"{pexels_root}delicious food", "/home/zhaowangbo/zangwei/minisora_new/pexels_delicious_food_fnt.csv"),
#     (f"{pexels_root}fashion", "/home/zhaowangbo/zangwei/minisora_new/pexels_fashion_fnt.csv"),
#     (f"{pexels_root}nature", "/home/zhaowangbo/zangwei/minisora_new/pexels_nature_fnt.csv"),
#     (f"{pexels_root}portrait", "/home/zhaowangbo/zangwei/minisora_new/pexels_portrait_fnt.csv"),
#     (
#         f"{pexels_root}science and technology",
#         "/home/zhaowangbo/zangwei/minisora_new/pexels_science_and_technology_fnt.csv",
#     ),
#     (f"{pexels_root}ship", "/home/zhaowangbo/zangwei/minisora_new/pexels_ship_fnt.csv"),
#     (f"{pexels_root}sports", "/home/zhaowangbo/zangwei/minisora_new/pexels_sports_fnt.csv"),
# )
infos = (
    ("", "preprocess/pexels_fnt.csv"),
    ("", "preprocess/inter4k_root_relength.csv"),
)
output = "pexels_inter4k.csv"


def main():
    data = []
    for path, csv_path in infos:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            data_ = list(reader)
            data += data_
        print("Number of videos in", path, ":", len(data_))
    print("Number of videos in total:", len(data))

    with open(output, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    print("Output saved to", output)


if __name__ == "__main__":
    main()

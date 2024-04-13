## Scene Detection and Video Splitting

### Formatting

Input meta should be `{prefix}.csv` with column `'videoId'`

```bash
python tools/scene_cut/process_meta.py --task append_format --meta_path /mnt/hdd/data/pexels_new/raw/meta/popular_6.csv --split popular_6
```

Output is `{prefix}_format.csv` (with column `path`) and `{prefix}_intact.csv` (with column `intact` and `path`)

### Scene Detection

Input meta should be `{prefix}_format.csv`

```bash
python tools/scene_cut/scene_detect.py --meta_path /mnt/hdd/data/pexels_new/raw/meta/popular_6_format.csv
```

Output is `{prefix}_format_timestamp.csv`

### Video Splitting

Input meta should be `{prefix}_timestamp.csv`

```bash
python tools/scene_cut/main_cut_pandarallel.py \
    --meta_path /mnt/hdd/data/pexels_new/raw/meta/popular_6_format_timestamp.csv \
    --out_dir /mnt/hdd/data/pexels_new/scene_cut/data/popular_6
```

Output is `{out_dir}/{wo_ext}_scene-{sid}.mp4`

TODO: meta for video clips

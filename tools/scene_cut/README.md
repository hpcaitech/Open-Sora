# Scene Detection and Video Splitting

- [Scene Detection and Video Splitting](#scene-detection-and-video-splitting)
    - [Prepare Meta Files](#prepare-meta-files)
    - [Scene Detection](#scene-detection)
    - [Video Splitting](#video-splitting)

In many cases, raw videos contain several scenes and are too long for training. Thus, it is essential to split them into shorter
clips based on scenes. Here, we provide code for scene detection and video splitting.

## Prepare Meta Files
At this step, you should have a raw video dataset prepared. A meta file of the dataset information is needed for data processing. To create a meta file from a folder, run:

```bash
python -m tools.datasets.convert video /path/to/video/folder --output /path/to/save/meta.csv
```
This should output a `.csv` file with column `path`.

If you already have a meta file for the videos and want to keep the information.
**Make sure** the meta file has column `id`, which is the id for each video, and the video is named as `{id}.mp4`.
The following command will add a new column `path` to the meta file.

```bash
python tools/scene_cut/convert_id_to_path.py /path/to/meta.csv --folder_path /path/to/video/folder
```
This should output
- `{prefix}_path-filtered.csv` with column `path` (broken videos filtered)
- `{prefix}_path_intact.csv` with column `path` and `intact` (`intact` indicating a video is intact or not)


## Scene Detection

Install the required dependancies by following our [installation instructions](../../docs/installation.md)'s "Data Dependencies" and "Scene Detection" sections.

<!-- The next step is to detect scenes in a video.
We use [`PySceneDetect`](https://github.com/Breakthrough/PySceneDetect) for this job.
```bash
pip install scenedetect[opencv] --upgrade
``` -->

**Make sure** the input meta file has column `path`, which is the path of a video.

```bash
python tools/scene_cut/scene_detect.py /path/to/meta.csv
```
The output is `{prefix}_timestamp.csv` with column `timestamp`. Each cell in column `timestamp` is a list of tuples,
with each tuple indicating the start and end timestamp of a scene
(e.g., `[('00:00:01.234', '00:00:02.345'), ('00:00:03.456', '00:00:04.567')]`).

## Video Splitting
After obtaining timestamps for scenes, we conduct video splitting (cutting).
**Make sure** the meta file contains column `timestamp`.

```bash
python tools/scene_cut/cut.py /path/to/meta.csv --save_dir /path/to/output/dir
```

This will save video clips to `/path/to/output/dir`. The video clips are named as `{video_id}_scene-{scene_id}.mp4`

To create a new meta file for the generated clips, run:
```bash
python -m tools.datasets.convert video /path/to/video/folder --output /path/to/save/meta.csv
```

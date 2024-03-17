# Scene Detection and Video Split

Raw videos from the Internet may be too long for training. 
Thus, we detect scenes in raw videos and split them into short clips based on the scenes.
First prepare the video processing packages.
```bash
pip install scenedetect moviepy opencv-python
```
Then run `scene_detect.py`. We provide efficient processing using `multiprocessing`. Don't forget to specify your own dataset path.

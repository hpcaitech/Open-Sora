# Data Processing
We establish a complete pipeline for video/image data processing. The pipeline is shown below.

![pipeline](/assets/readme/report_data_pipeline.png)

First, raw videos, 
either from the  Internet or public datasets, are split into shorter clips based on scene detection. 
Then, we evaluate these videos by predicting multiple scores using existing models. We first predict the aesthetic score 
and the optical flow score for a video. We also conduct OCR to detect texts in the video. Only videos with satisfactory 
evaluation results are sent to the next step for captioning. After captioning, the matching score is also calculated as 
an assessment of video-text alignment. Finally, we filter samples based on the matching score and 
conduct camera motion detection for the remaining samples. 
In summary, our pipeline produces video-text pairs which have high aesthetic quality, large video motion and strong 
semantic consistency.

For more information, please refer to:
- [Dataset Management](https://github.com/hpcaitech/Open-Sora-dev/blob/dev/v1.1/tools/datasets/README.md)
- [Scene Detection and Video Splitting](https://github.com/hpcaitech/Open-Sora-dev/blob/dev/v1.1/tools/scene_cut/README.md)
- [Scoring and Filtering](https://github.com/hpcaitech/Open-Sora-dev/blob/dev/v1.1/tools/scoring/README.md)
- [Captioning](https://github.com/hpcaitech/Open-Sora-dev/blob/dev/v1.1/tools/caption/README.md)
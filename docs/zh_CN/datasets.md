# 数据集

## 正在使用的数据集

### HD-VG-130M

[HD-VG-130M](https://github.com/daooshee/HD-VG-130M?tab=readme-ov-file) 包括 130M 个文本视频对。标题是
由 BLIP-2 生成。我们发现剪切和文本质量相对较差。它包含 20 个拆分。对于 OpenSora 1.0，我们使用第一个拆分。我们计划使用整个数据集并对其进行重新处理。

### Inter4k

[Inter4k](https://github.com/alexandrosstergiou/Inter4K) 是一个包含分辨率为 4K 的 1k 视频剪辑的数据集。这个
数据集被提议用于超分辨率任务。我们使用数据集进行 HQ 训练。处理过的视频可以从这里找到 [这里](README.md#数据处理) 。

### Pexels.com

[Pexels.com](https://www.pexels.com/) 是一个提供免费库存照片和视频的网站。我们收集的 19K 视频
来自本网站的剪辑，用于高质量训练。处理过的视频可以从这里找到 [这里](README.md#数据处理) 。

## 数据集监视列表

我们也在关注以下数据集，并考虑在未来使用它们，这取决于我们的存储空间以及数据集的质量。

| 名称                | 大小           | 描述                            |
|-------------------|--------------|-------------------------------|
| Panda-70M         | 70M videos   | High quality video-text pairs |
| WebVid-10M        | 10M videos   | Low quality                   |
| InternVid-10M-FLT | 10M videos   |                               |
| EGO4D             | 3670 hours   |                               |
| OpenDV-YouTube    | 1700 hours   |                               |
| VidProM           | 6.69M videos |                               |

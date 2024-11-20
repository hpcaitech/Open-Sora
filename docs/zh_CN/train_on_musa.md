# 在MUSA环境训练

## 数据预处理

### Step 1.1 准备数据

创建用于存放视频(video), 片段(clips)和视频处理信息的文件夹(meta);
```bash
mkdir ./dataset/panda3m
mkdir ./dataset/panda3m/video
mkdir ./dataset/panda3m/clips
mkdir ./dataset/panda3m/meta
```

设置目录相应的环境变量
```bash
export ROOT_VIDEO="./dataset/panda3m/video" 
export ROOT_CLIPS="./dataset/panda3m/clips" 
export ROOT_META="./dataset/panda3m/meta"
```

### Step 1.2 创建meta文件

从video创建meta file; meta文件应该在${ROOT_META}/meta.csv；
运行命令: 
```bash
python -m tools.datasets.convert video ${ROOT_VIDEO} --output ${ROOT_META}/meta.csv
```

### Step 1.3 整理视频文件

获取文件信息，删除损坏视频；输出存放至${ROOT_META}/meta_info_fmin1.csv
```bash
python -m tools.datasets.datautil ${ROOT_META}/meta.csv --info --fmin 1
```

### Step 2.1 场景检测

输出存放至${ROOT_META}/meta_info_fmin1_timestamp.csv; 完成场景检测后，处理完成的meta_info_fmin1.csv包含path,num_frames,height,width,aspect_ratio,fps,resolution信息
```bash
python -m tools.scene_cut.scene_detect ${ROOT_META}/meta_info_fmin1.csv
```

### Step 2.2 基于场景的视频片段切分

片段输出存放至${ROOT_CLIPS}; 片段信息存放至${ROOT_META}/meta_info_fmin1_timestamp.csv， 信息包含包含path,num_frames,height,width,aspect_ratio,fps,resolution,timestamp
```bash
python -m tools.scene_cut.cut ${ROOT_META}/meta_info_fmin1_timestamp.csv --save_dir ${ROOT_CLIPS}
```

### Step 2.3 创建关于片段的meta文件

创建关于片段的meta文件，meta文件输出存放至${ROOT_META}/meta_clips.csv
```bash
python -m tools.datasets.convert video ${ROOT_CLIPS} --output ${ROOT_META}/meta_clips.csv
```

### Step 2.4 整理片段文件

获取片段信息，删除损坏片段； 整合后的片段信息存放至${ROOT_META}/meta_clips_info_fmin1.csv
```bash
python -m tools.datasets.datautil ${ROOT_META}/meta_clips.csv --info --fmin 1
```

### Step 3.1 预测美学评分

使用预训练的模型aesthetic-model预测片段的美学评分，输出存放至{ROOT_META}/meta_clips_info_fmin1_aes_part*.csv 
```bash
torchrun --nproc_per_node 8 -m tools.scoring.aesthetic.inference 
  ${ROOT_META}/meta_clips_info_fmin1.csv 
  --bs 1024 
  --num_workers 16
```

### Step 3.2 整合预测美学评分结果

如果Step 3.1中使用了分布式预测，本步骤会将整合多个{ROOT_META}/meta_clips_info_fmin1_aes_part*.csv 至一个{ROOT_META}/meta_clips_info_fmin1_aes.csv
```bash
python -m tools.datasets.datautil ${ROOT_META}/meta_clips_info_fmin1_aes_part*.csv --output ${ROOT_META}/meta_clips_info_fmin1_aes.csv
```

### Step 3.3 根据美学评分将片段聚类

输出存放至${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5.csv
```bash
python -m tools.datasets.datautil ${ROOT_META}/meta_clips_info_fmin1_aes.csv --aesmin 5
```

### Step 4.1 生成说明文本

基于llava生成视频的说明文字; 输出存放至${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5_caption_part*.csv
```bash
torchrun --nproc_per_node 4 --standalone -m tools.caption.caption_llava 
  ${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5.0.csv 
  --dp-size 4 
  --tp-size 1 
  --model-path ./pretrained_models/llava/ 
  --prompt video
```

### Step 4.2 整合视频说明文本

若Step 4.1使用了分布式预测，本步骤会将整合多个${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5_caption_part*.csv至一个${ROOT_META}/meta_clips_caption.csv
```bash
python -m tools.datasets.datautil ${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5.0_caption_part*.csv --output ${ROOT_META}/meta_clips_caption.csv
```

### Step 4.3 整理视频说明文本
根据T5 pipline 修改标题，以适应训练需要；修改LLM生成的文本；情况行为空的说明文本；
```bash
python -m tools.datasets.datautil 
  ${ROOT_META}/meta_clips_caption.csv 
  --clean-caption 
  --refine-llm-caption 
  --remove-empty-caption 
  --output ${ROOT_META}/meta_clips_caption_cleaned.csv
```

## 训练

### 单机单卡
```bash
torchrun --nnodes=1 --nproc_per_node=1 scripts/train.py configs/opensora/train/16x256x256.py --data-path ./dataset/panda2m/meta/meta_clips_caption_cleaned.csv
```
### 单机多卡
```bash
torchrun --nnodes=1 --nproc_per_node=4 scripts/train.py configs/opensora/train/16x256x256.py --data-path ./dataset/panda2m/meta/meta_clips_caption_cleaned.csv
```


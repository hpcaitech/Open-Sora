import pandas as pd
import os

data_path = '/home/dist/hpcai/duanjunwen/Open-Sora/dataset/panda_train_2/meta/meta_clips_caption_cleaned_fixed_rm4_format.csv'
good_data_path = '/home/dist/hpcai/duanjunwen/Open-Sora/dataset/panda_train_2/meta/meta_clips_caption_cleaned_fixed_rm5.csv'
# remove bad sample from dataset
good_df_dict = {
    "path":[],
    "text":[],
    "num_frames":[],
    "height":[],
    "width":[],
    "aspect_ratio":[],
    'fps':[],
    "resolution":[]
}
bad_samples = ['00001298_scene-28.mp4', '00001325_scene-1.mp4', '00001221_scene-0.mp4', '00001210_scene-59.mp4', '00001386_scene-7.mp4', '00001341_scene-12.mp4', '00001315_scene-1.mp4', '00001399_scene-15.mp4', '00001221_scene-13.mp4', '00001367_scene-21.mp4', '00001325_scene-188.mp4', '00001325_scene-161.mp4', '00001325_scene-193.mp4', '00001247_scene-68.mp4', '00001210_scene-41.mp4', '00001297_scene-3.mp4', '00001298_scene-28.mp4', '00001325_scene-1.mp4', '00001221_scene-0.mp4', '00001210_scene-59.mp4', '00001386_scene-7.mp4', '00001341_scene-12.mp4', '00001315_scene-1.mp4', '00001399_scene-15.mp4', '00001221_scene-13.mp4', '00001367_scene-21.mp4', '00001325_scene-188.mp4', '00001325_scene-161.mp4', '00001325_scene-193.mp4', '00001247_scene-68.mp4', '00001210_scene-41.mp4', '00001297_scene-3.mp4', '00001298_scene-28.mp4', '00001325_scene-1.mp4', '00001221_scene-0.mp4', '00001210_scene-59.mp4', '00001386_scene-7.mp4', '00001341_scene-12.mp4', '00001315_scene-1.mp4', '00001399_scene-15.mp4', '00001298_scene-28.mp4', '00001325_scene-1.mp4', '00001221_scene-0.mp4', '00001210_scene-59.mp4', '00001386_scene-7.mp4', '00001341_scene-12.mp4', '00001315_scene-1.mp4', '00001399_scene-15.mp4', '00001221_scene-13.mp4', '00001367_scene-21.mp4', '00001325_scene-188.mp4', '00001325_scene-161.mp4', '00001325_scene-193.mp4', '00001247_scene-68.mp4', '00001210_scene-41.mp4', '00001297_scene-3.mp4', '00001221_scene-13.mp4', '00001367_scene-21.mp4', '00001325_scene-188.mp4', '00001325_scene-161.mp4', '00001325_scene-193.mp4', '00001247_scene-68.mp4', '00001210_scene-41.mp4', '00001297_scene-3.mp4',
            '00001219_scene-21.mp4', '00001217_scene-103.mp4', '00001214_scene-41.mp4', '00001397_scene-34.mp4', '00001380_scene-57.mp4', '00001233_scene-29.mp4', '00001201_scene-67.mp4', '00001325_scene-107.mp4', '00001325_scene-192.mp4', '00001210_scene-7.mp4', '00001372_scene-61.mp4', '00001201_scene-29.mp4', '00001325_scene-148.mp4', '00001294_scene-131.mp4', '00001219_scene-0.mp4', '00001281_scene-2.mp4', '00001219_scene-21.mp4', '00001217_scene-103.mp4', '00001214_scene-41.mp4', '00001397_scene-34.mp4', '00001380_scene-57.mp4', '00001233_scene-29.mp4', '00001201_scene-67.mp4', '00001325_scene-107.mp4', '00001325_scene-192.mp4', '00001210_scene-7.mp4', '00001372_scene-61.mp4', '00001201_scene-29.mp4', '00001325_scene-148.mp4', '00001294_scene-131.mp4', '00001219_scene-0.mp4', '00001281_scene-2.mp4', '00001219_scene-21.mp4', '00001217_scene-103.mp4', '00001214_scene-41.mp4', '00001397_scene-34.mp4', '00001380_scene-57.mp4', '00001233_scene-29.mp4', '00001201_scene-67.mp4', '00001325_scene-107.mp4', '00001219_scene-21.mp4', '00001217_scene-103.mp4', '00001214_scene-41.mp4', '00001397_scene-34.mp4', '00001380_scene-57.mp4', '00001233_scene-29.mp4', '00001201_scene-67.mp4', '00001325_scene-107.mp4', '00001325_scene-192.mp4', '00001210_scene-7.mp4', '00001372_scene-61.mp4', '00001201_scene-29.mp4', '00001325_scene-148.mp4', '00001294_scene-131.mp4', '00001219_scene-0.mp4', '00001281_scene-2.mp4', '00001325_scene-192.mp4', '00001210_scene-7.mp4', '00001372_scene-61.mp4', '00001201_scene-29.mp4', '00001325_scene-148.mp4', '00001294_scene-131.mp4', '00001219_scene-0.mp4', '00001281_scene-2.mp4'] 
bad_samples2 = ['00001298_scene-28.mp4', '00001325_scene-1.mp4', '00001221_scene-0.mp4', '00001210_scene-59.mp4', '00001386_scene-7.mp4', '00001341_scene-12.mp4', '00001315_scene-1.mp4', '00001399_scene-15.mp4', '00001221_scene-13.mp4', '00001367_scene-21.mp4', '00001325_scene-188.mp4', '00001325_scene-161.mp4', '00001325_scene-193.mp4', '00001247_scene-68.mp4', '00001210_scene-41.mp4', '00001297_scene-3.mp4', '00001298_scene-28.mp4', '00001325_scene-1.mp4', '00001221_scene-0.mp4', '00001210_scene-59.mp4', '00001386_scene-7.mp4', '00001341_scene-12.mp4', '00001315_scene-1.mp4', '00001399_scene-15.mp4', '00001221_scene-13.mp4', '00001367_scene-21.mp4', '00001325_scene-188.mp4', '00001325_scene-161.mp4', '00001325_scene-193.mp4', '00001247_scene-68.mp4', '00001210_scene-41.mp4', '00001297_scene-3.mp4', '00001298_scene-28.mp4', '00001325_scene-1.mp4', '00001221_scene-0.mp4', '00001210_scene-59.mp4', '00001386_scene-7.mp4', '00001341_scene-12.mp4', '00001315_scene-1.mp4', '00001399_scene-15.mp4', '00001298_scene-28.mp4', '00001325_scene-1.mp4', '00001221_scene-0.mp4', '00001210_scene-59.mp4', '00001386_scene-7.mp4', '00001341_scene-12.mp4', '00001315_scene-1.mp4', '00001399_scene-15.mp4', '00001221_scene-13.mp4', '00001367_scene-21.mp4', '00001325_scene-188.mp4', '00001325_scene-161.mp4', '00001325_scene-193.mp4', '00001247_scene-68.mp4', '00001210_scene-41.mp4', '00001297_scene-3.mp4', '00001221_scene-13.mp4', '00001367_scene-21.mp4', '00001325_scene-188.mp4', '00001325_scene-161.mp4', '00001325_scene-193.mp4', '00001247_scene-68.mp4', '00001210_scene-41.mp4', '00001297_scene-3.mp4',
            '00001219_scene-21.mp4', '00001217_scene-103.mp4', '00001214_scene-41.mp4', '00001397_scene-34.mp4', '00001380_scene-57.mp4', '00001233_scene-29.mp4', '00001201_scene-67.mp4', '00001325_scene-107.mp4', '00001325_scene-192.mp4', '00001210_scene-7.mp4', '00001372_scene-61.mp4', '00001201_scene-29.mp4', '00001325_scene-148.mp4', '00001294_scene-131.mp4', '00001219_scene-0.mp4', '00001281_scene-2.mp4', '00001219_scene-21.mp4', '00001217_scene-103.mp4', '00001214_scene-41.mp4', '00001397_scene-34.mp4', '00001380_scene-57.mp4', '00001233_scene-29.mp4', '00001201_scene-67.mp4', '00001325_scene-107.mp4', '00001325_scene-192.mp4', '00001210_scene-7.mp4', '00001372_scene-61.mp4', '00001201_scene-29.mp4', '00001325_scene-148.mp4', '00001294_scene-131.mp4', '00001219_scene-0.mp4', '00001281_scene-2.mp4', '00001219_scene-21.mp4', '00001217_scene-103.mp4', '00001214_scene-41.mp4', '00001397_scene-34.mp4', '00001380_scene-57.mp4', '00001233_scene-29.mp4', '00001201_scene-67.mp4', '00001325_scene-107.mp4', '00001219_scene-21.mp4', '00001217_scene-103.mp4', '00001214_scene-41.mp4', '00001397_scene-34.mp4', '00001380_scene-57.mp4', '00001233_scene-29.mp4', '00001201_scene-67.mp4', '00001325_scene-107.mp4', '00001325_scene-192.mp4', '00001210_scene-7.mp4', '00001372_scene-61.mp4', '00001201_scene-29.mp4', '00001325_scene-148.mp4', '00001294_scene-131.mp4', '00001219_scene-0.mp4', '00001281_scene-2.mp4', '00001325_scene-192.mp4', '00001210_scene-7.mp4', '00001372_scene-61.mp4', '00001201_scene-29.mp4', '00001325_scene-148.mp4', '00001294_scene-131.mp4', '00001219_scene-0.mp4', '00001281_scene-2.mp4'] 

for i in range(len(bad_samples)):
    bad_samples[i] = "/home/dist/hpcai/duanjunwen/Open-Sora/dataset/panda_train/clips/" + bad_samples[i]
    bad_samples2[i] = "/home/dist/hpcai/duanjunwen/Open-Sora/dataset/panda_train_2/clips/" + bad_samples2[i]

bad_df = pd.read_csv(data_path)    
datashape = bad_df.shape
for idx, row in bad_df.iterrows():
    # print(row['path'] not in bad_samples)
    # print(row['path'] not in bad_samples and row['path'] not in bad_samples2)
    if row['path'] not in bad_samples and row['path'] not in bad_samples2:
        good_df_dict["path"].append(row['path'])
        good_df_dict["text"].append(row['text'])
        good_df_dict["num_frames"].append(row['num_frames'])
        good_df_dict["height"].append(row['height'])
        good_df_dict["width"].append(row['width'])
        good_df_dict["aspect_ratio"].append(row['aspect_ratio'])
        good_df_dict["fps"].append(row['fps'])
        good_df_dict["resolution"].append(row['resolution'])
            
good_df = pd.DataFrame(columns=["path","text","num_frames","height","width","aspect_ratio",'fps',"resolution"])
good_df["path"] = good_df_dict['path']
good_df["text"] = good_df_dict['text']
good_df["num_frames"] = good_df_dict['num_frames']
good_df["height"] = good_df_dict['height']
good_df["width"] = good_df_dict['width']
good_df["aspect_ratio"] = good_df_dict['aspect_ratio']
good_df['fps'] = good_df_dict['fps']
good_df["resolution"] = good_df_dict['resolution']
good_df.drop_duplicates()
print(f"datashape {bad_df.shape}; datashape {good_df.shape}")
good_df.to_csv(good_data_path, index=False)

# fix data_path, good_data_path
# python scripts/remove_bad_data.py 
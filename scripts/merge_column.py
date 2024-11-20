import pandas as pd
import os

def replace_func(x):
    bad_file_list = [] 
    for filename in x.split(','):
        filename = filename.strip('[').strip(']').split('/')[-1].strip('\'')
        bad_file_list.append(filename)
   
    return bad_file_list
dir_path = './loss_curve/musa_merge14/'
file_list = os.listdir("/home/dist/hpcai/duanjunwen/Open-Sora/loss_curve/musa_merge14")

df_list = []
for file in file_list:
    df = pd.read_csv(dir_path + file)        
    df['path'] = df['path'].apply(replace_func)
    df_list.append(df)

base_df =df_list[0]

for curr_df in df_list[1:]:
    base_df['path'] = base_df['path'] + curr_df['path']

base_df.to_csv(f"./loss_curve/musa_merge14/musa_loss_curve_with_path_merged.csv", index=False)

# fix musa_merge14
# python scripts/merge_column.py       
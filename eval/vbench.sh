#!/bin/bash

# Base path for videos
videos_path='./samples/samples_stage1_41k_2_17k_vbench'
json_path='./eval/VBench_full_info.json'

# Define the dimension list
dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "motion_smoothness" "dynamic_degree" "appearance_style")

# Loop over each dimension
for i in "${!dimensions[@]}"; do
    # Get the dimension and corresponding folder
    dimension=${dimensions[i]}

    # Construct the video path
    echo "$dimension $videos_path"

    # Run the evaluation script
    vbench evaluate --videos_path $videos_path --dimension $dimension --full_json_dir $json_path
done

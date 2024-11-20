#!/bin/bash

set -e

# Base path for videos
videos_path=$1
videos_base=$(basename $videos_path)
json_path=./eval/vbench/VBench_full_info.json
output_path=./evaluation_results/$videos_base

# Define the dimension list
dimensions=(
    # Quality Score
    "subject_consistency"
    "background_consistency"
    "motion_smoothness"
    "dynamic_degree"
    "aesthetic_quality"
    "imaging_quality"
    "temporal_flickering"
    # Semantic Score
    "object_class"
    "multiple_objects"
    "color"
    "spatial_relationship"
    "scene"
    "temporal_style"
    "overall_consistency"
    "human_action"
    "appearance_style"
)

# Loop over each dimension
for i in "${!dimensions[@]}"; do
    # Get the dimension and corresponding folder
    dimension=${dimensions[i]}

    # Construct the video path
    echo "$dimension $videos_path"

    # Run the evaluation script
    vbench evaluate --videos_path $videos_path --dimension $dimension --full_json_dir $json_path --output_path $output_path
done

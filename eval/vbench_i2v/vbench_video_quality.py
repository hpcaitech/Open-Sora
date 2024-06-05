from vbench import VBench


VIDEO_PATH = ""
DIMENSIONS = ["subject_consistency", "background_consistency", "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality", "temporal_flickering"]

my_VBench = VBench("cuda", "vbench2_beta_i2v/vbench2_i2v_full_info.json", "evaluation_results")
my_VBench.evaluate(
    videos_path = VIDEO_PATH,
    name = "vbench_video_quality",
    dimension_list = DIMENSIONS,
)
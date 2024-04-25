from vbench2_beta_i2v import VBenchI2V

VIDEO_PATH = ""
DIMENSIONS = ["i2v_subject", "i2v_background", "camera_motion"]

my_VBench = VBenchI2V("cuda", "vbench2_beta_i2v/vbench2_i2v_full_info.json", "evaluation_results")
my_VBench.evaluate(videos_path=VIDEO_PATH, name="vbench_i2v", dimension_list=DIMENSIONS, resolution="1-1")

import os

import numpy as np
import torch

from .utils import load_video
from .visualizer import Visualizer


def transform(vector):
    x = np.mean([item[0] for item in vector])
    y = np.mean([item[1] for item in vector])
    return [x, y]


class CameraPredict:
    def __init__(self, device, submodules_list, factor=0.25):
        self.device = device
        self.grid_size = 10
        self.factor = factor
        try:
            self.model = torch.hub.load(submodules_list["repo"], submodules_list["model"]).to(self.device)
        except:
            # workaround for CERTIFICATE_VERIFY_FAILED (see: https://github.com/pytorch/pytorch/issues/33288#issuecomment-954160699)
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context
            self.model = torch.hub.load(submodules_list["repo"], submodules_list["model"]).to(self.device)

    def infer(self, video_path, save_video=False, save_dir="./saved_videos"):
        # load video
        video = load_video(video_path, return_tensor=False)
        # set scale
        height, width = video.shape[1], video.shape[2]
        self.scale = min(height, width)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(self.device)  # B T C H W
        pred_tracks, pred_visibility = self.model(video, grid_size=self.grid_size)  # B T N 2,  B T N 1

        if save_video:
            video_name = os.path.basename(video_path)[:-4]
            vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=3)
            vis.visualize(video, pred_tracks, pred_visibility, filename=video_name)

        return pred_tracks[0].long().detach().cpu().numpy()

    def transform_class(self, vector, min_reso):  # 768*0.05
        scale = min_reso * self.factor
        x, y = vector
        direction = []
        if x > scale:
            direction.append("right")
        elif x < -scale:
            direction.append("left")

        if y > scale:
            direction.append("down")
        elif y < -scale:
            direction.append("up")

        return direction if direction else ["static"]

    def get_edge_point(self, track):
        middle = self.grid_size // 2
        top = [list(track[0, i, :]) for i in range(middle - 2, middle + 2)]
        down = [list(track[self.grid_size - 1, i, :]) for i in range(middle - 2, middle + 2)]
        left = [list(track[i, 0, :]) for i in range(middle - 2, middle + 2)]
        right = [list(track[i, self.grid_size - 1, :]) for i in range(middle - 2, middle + 2)]

        return top, down, left, right

    def get_edge_direction(self, track1, track2):
        edge_points1 = self.get_edge_point(track1)
        edge_points2 = self.get_edge_point(track2)

        vector_results = []
        for points1, points2 in zip(edge_points1, edge_points2):
            vectors = [[end[0] - start[0], end[1] - start[1]] for start, end in zip(points1, points2)]
            vector_results.append(vectors)
        vector_results = list(map(transform, vector_results))
        class_results = [self.transform_class(vector, min_reso=self.scale) for vector in vector_results]

        return class_results

    def classify_top_down(self, top, down):
        results = []
        classes = [f"{item_t}_{item_d}" for item_t in top for item_d in down]

        results_mapping = {
            "left_left": "pan_right",
            "right_right": "pan_left",
            "down_down": "tilt_up",
            "up_up": "tilt_down",
            "up_down": "zoom_in",
            "down_up": "zoom_out",
            "static_static": "static",
        }
        results = [results_mapping.get(cls) for cls in classes if cls in results_mapping]
        return results if results else ["None"]

    def classify_left_right(self, left, right):
        results = []
        classes = [f"{item_l}_{item_r}" for item_l in left for item_r in right]
        results_mapping = {
            "left_left": "pan_right",
            "right_right": "pan_left",
            "down_down": "tilt_up",
            "up_up": "tilt_down",
            "left_right": "zoom_in",
            "right_left": "zoom_out",
            "static_static": "static",
        }
        results = [results_mapping.get(cls) for cls in classes if cls in results_mapping]
        return results if results else ["None"]

    def camera_classify(self, track1, track2):
        top, down, left, right = self.get_edge_direction(track1, track2)

        top_results = self.classify_top_down(top, down)
        left_results = self.classify_left_right(left, right)

        results = list(set(top_results + left_results))
        if "None" in results and len(results) > 1:
            results.remove("None")
        if "static" in results and len(results) > 1:
            results.remove("static")
        if len(results) == 1 and results[0] == "None":  # Tom added this to deal with edge cases
            results = ["Undetermined"]
        return results

    def predict(self, video_path):
        pred_track = self.infer(video_path)
        track1 = pred_track[0].reshape((self.grid_size, self.grid_size, 2))
        track2 = pred_track[-1].reshape((self.grid_size, self.grid_size, 2))
        results = self.camera_classify(track1, track2)
        return results


def compute_camera_motion(device, submodules_dict, video_paths, factor):
    camera = CameraPredict(device, submodules_dict, factor)
    # predict_results = camera.predict(video_path)
    # return predict_results
    all_predictions = []
    for video_path in video_paths:
        camera_motion_types = camera.predict(video_path)
        all_predictions.append("+".join(camera_motion_types))
    return all_predictions

from collections import OrderedDict

import torch

from .aspect import ASPECT_RATIOS, get_closest_ratio


def find_approximate_hw(hw, hw_dict, approx=0.8):
    for k, v in hw_dict.items():
        if hw >= v * approx:
            return k
    return None


def find_closet_smaller_bucket(t, t_dict, frame_interval):
    # process image
    if t == 1:
        if 1 in t_dict:
            return 1
        else:
            return None
    # process video
    for k, v in t_dict.items():
        if t >= v * frame_interval and v != 1:
            return k
    return None


class Bucket:
    def __init__(self, bucket_config):
        for key in bucket_config:
            assert key in ASPECT_RATIOS, f"Aspect ratio {key} not found."
        # wrap config with OrderedDict
        bucket_probs = OrderedDict()
        bucket_bs = OrderedDict()
        bucket_names = sorted(bucket_config.keys(), key=lambda x: ASPECT_RATIOS[x][0], reverse=True)
        for key in bucket_names:
            bucket_time_names = sorted(bucket_config[key].keys(), key=lambda x: x, reverse=True)
            bucket_probs[key] = OrderedDict({k: bucket_config[key][k][0] for k in bucket_time_names})
            bucket_bs[key] = OrderedDict({k: bucket_config[key][k][1] for k in bucket_time_names})

        # first level: HW
        num_bucket = 0
        hw_criteria = dict()
        t_criteria = dict()
        ar_criteria = dict()
        for k1, v1 in bucket_probs.items():
            hw_criteria[k1] = ASPECT_RATIOS[k1][0]
            t_criteria[k1] = dict()
            ar_criteria[k1] = dict()
            for k2, _ in v1.items():
                t_criteria[k1][k2] = k2
                ar_criteria[k1][k2] = dict()
                for k3, v3 in ASPECT_RATIOS[k1][1].items():
                    ar_criteria[k1][k2][k3] = v3
                    num_bucket += 1

        self.bucket_probs = bucket_probs
        self.bucket_bs = bucket_bs
        self.hw_criteria = hw_criteria
        self.t_criteria = t_criteria
        self.ar_criteria = ar_criteria
        self.num_bucket = num_bucket
        print(f"Number of buckets: {num_bucket}")

    def info_bucket(self, dataset, frame_interval=1):
        infos = dict()
        infos_ar = dict()
        for i in range(len(dataset)):
            T, H, W = dataset.get_data_info(i)
            bucket_id = self.get_bucket_id(T, H, W, frame_interval)
            if bucket_id is None:
                continue
            if f"{(bucket_id[0], bucket_id[1])}" not in infos:
                infos[f"{(bucket_id[0], bucket_id[1])}"] = 0
            if f"{bucket_id[2]}" not in infos_ar:
                infos_ar[f"{bucket_id[2]}"] = 0
            infos[f"{(bucket_id[0], bucket_id[1])}"] += 1
            infos_ar[f"{bucket_id[2]}"] += 1
        print(f"Dataset contains {len(dataset)} samples.")
        print("Bucket info:", infos)
        print("Aspect ratio info:", infos_ar)

    def get_bucket_id(self, T, H, W, frame_interval=1, generator=None):
        # hw
        hw = H * W
        hw_id = find_approximate_hw(hw, self.hw_criteria)
        if hw_id is None:
            return None
        hw_id_index = list(self.hw_criteria.keys()).index(hw_id)

        # hw drops by probablity
        while True:
            # T
            T_id = find_closet_smaller_bucket(T, self.t_criteria[hw_id], frame_interval)
            if T_id is not None:
                prob = self.get_prob((hw_id, T_id))
                if torch.rand(1, generator=generator).item() < prob:
                    break
            hw_id_index += 1
            if hw_id_index > len(self.hw_criteria) - 1:
                break
            hw_id = list(self.hw_criteria.keys())[hw_id_index]

        if T_id is None or hw_id_index > len(self.hw_criteria) - 1:
            return None

        # ar
        ar_criteria = self.ar_criteria[hw_id][T_id]
        ar_id = get_closest_ratio(H, W, ar_criteria)
        return hw_id, T_id, ar_id

    def get_thw(self, bucket_id):
        assert len(bucket_id) == 3
        T = self.t_criteria[bucket_id[0]][bucket_id[1]]
        H, W = self.ar_criteria[bucket_id[0]][bucket_id[1]][bucket_id[2]]
        return T, H, W

    def get_prob(self, bucket_id):
        return self.bucket_probs[bucket_id[0]][bucket_id[1]]

    def get_batch_size(self, bucket_id):
        return self.bucket_bs[bucket_id[0]][bucket_id[1]]

    def __len__(self):
        return self.num_bucket


def closet_smaller_bucket(value, bucket):
    for i in range(1, len(bucket)):
        if value < bucket[i]:
            return bucket[i - 1]
    return bucket[-1]

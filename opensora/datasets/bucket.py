import bisect
from collections import OrderedDict

import numpy as np

from opensora.utils.misc import get_logger

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
        bucket_id = OrderedDict()
        bucket_id_cnt = 0
        for k1, v1 in bucket_probs.items():
            hw_criteria[k1] = ASPECT_RATIOS[k1][0]
            t_criteria[k1] = dict()
            ar_criteria[k1] = dict()
            bucket_id[k1] = dict()
            for k2, _ in v1.items():
                t_criteria[k1][k2] = k2
                bucket_id[k1][k2] = bucket_id_cnt
                bucket_id_cnt += 1
                ar_criteria[k1][k2] = dict()
                for k3, v3 in ASPECT_RATIOS[k1][1].items():
                    ar_criteria[k1][k2][k3] = v3
                    num_bucket += 1

        self.bucket_probs = bucket_probs
        self.bucket_bs = bucket_bs
        self.bucket_id = bucket_id
        self.hw_criteria = hw_criteria
        self.t_criteria = t_criteria
        self.ar_criteria = ar_criteria
        self.num_bucket = num_bucket
        get_logger().info("Number of buckets: %s", num_bucket)

    def get_bucket_id(self, T, H, W, frame_interval=1, seed=None):
        resolution = H * W
        approx = 0.8

        fail = True
        for hw_id, t_criteria in self.bucket_probs.items():
            if resolution < self.hw_criteria[hw_id] * approx:
                continue

            # if sample is an image
            if T == 1:
                if 1 in t_criteria:
                    rng = np.random.default_rng(seed + self.bucket_id[hw_id][1])
                    if rng.random() < t_criteria[1]:
                        fail = False
                        t_id = 1
                        break
                else:
                    continue

            # otherwise, find suitable t_id for video
            t_fail = True
            for t_id, prob in t_criteria.items():
                rng = np.random.default_rng(seed + self.bucket_id[hw_id][t_id])
                if isinstance(prob, tuple):
                    prob_t = prob[1]
                    if rng.random() > prob_t:
                        continue
                if T > t_id * frame_interval and t_id != 1:
                    t_fail = False
                    break
            if t_fail:
                continue

            # leave the loop if prob is high enough
            if isinstance(prob, tuple):
                prob = prob[0]
            if prob >= 1 or rng.random() < prob:
                fail = False
                break
        if fail:
            return None

        # get aspect ratio id
        ar_criteria = self.ar_criteria[hw_id][t_id]
        ar_id = get_closest_ratio(H, W, ar_criteria)
        return hw_id, t_id, ar_id

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


def merge_dicts(dict_list, maps):
    result = {}
    for index, d in enumerate(dict_list):
        for key, value in d.items():
            # Create the key with an empty set if it doesn't exist
            if key not in result:
                result[key] = []
            # Add the value and the tuple (value, index) to the set
            if value not in result[key]:
                result[key].append((value, maps[index]))
    return result


def find_proper_res(x1, y1, points):
    result_point = []
    points = sorted(points, key=lambda x: -x[0][0])
    for k, res in points:
        x2 = k[0]
        y2 = k[1]
        if x2 <= x1 and y2 <= y1:
            result_point.append(res)
    return result_point


class Bucket_ar_first(Bucket):
    def __init__(self, bucket_config):
        super(Bucket_ar_first, self).__init__(bucket_config)
        all_aspects = []
        maps = []
        for x in bucket_config.keys():
            maps.append(x)
            all_aspects.append(ASPECT_RATIOS[x][1])
        all_aspects = merge_dicts(all_aspects, maps)
        all_ars = [float(i) for i in all_aspects.keys()]

        self.all_aspects = all_aspects
        self.all_ars = sorted(all_ars)

    def get_bucket_id(self, T, H, W, frame_interval=1, seed=None):
        ar = H / W

        # binary search
        ind = bisect.bisect_left(self.all_ars, ar)
        if ind == 0:
            ar_id = str(self.all_ars[0])
        if ind == len(self.all_ars):
            ar_id = str(self.all_ars[-1])
        else:
            before = self.all_ars[ind - 1]
            after = self.all_ars[ind]
            ar_id = f"{before:.2f}" if abs(ar - before) < abs(ar - after) else f"{after:.2f}"

        # find proper hw_id
        hw_ids = find_proper_res(H, W, self.all_aspects[ar_id])
        fail = True
        for hw_id in hw_ids:
            t_fail = True
            for t_id, prob in self.bucket_probs[hw_id].items():
                if isinstance(prob, tuple):
                    rng = np.random.default_rng(seed + self.bucket_id[hw_id][t_id])
                    prob_t = prob[1]
                    if rng.random() > prob_t:
                        continue
                if T > t_id * frame_interval and t_id != 1:
                    t_fail = False
                    break
            if t_fail:
                continue

            if isinstance(prob, tuple):
                prob = prob[0]
            if prob >= 1 or rng.random() < prob:
                fail = False
                break
        if fail:
            return None

        return hw_id, t_id, ar_id

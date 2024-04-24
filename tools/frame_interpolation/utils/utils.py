import random
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from imageio import imread, imwrite
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterGroups:
    def __init__(self) -> None:
        self.meter_dict = dict()

    def update(self, dict, n=1):
        for name, val in dict.items():
            if self.meter_dict.get(name) is None:
                self.meter_dict[name] = AverageMeter()
            self.meter_dict[name].update(val, n)

    def reset(self, name=None):
        if name is None:
            for v in self.meter_dict.values():
                v.reset()
        else:
            meter = self.meter_dict.get(name)
            if meter is not None:
                meter.reset()

    def avg(self, name):
        meter = self.meter_dict.get(name)
        if meter is not None:
            return meter.avg


class InputPadder:
    """Pads images such that dimensions are divisible by divisor"""

    def __init__(self, dims, divisor=16):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    def pad(self, *inputs):
        if len(inputs) == 1:
            return F.pad(inputs[0], self._pad, mode="replicate")
        else:
            return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, *inputs):
        if len(inputs) == 1:
            return self._unpad(inputs[0])
        else:
            return [self._unpad(x) for x in inputs]

    def _unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


def img2tensor(img):
    if img.shape[-1] > 3:
        img = img[:, :, :3]
    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0) / 255.0


def tensor2img(img_t):
    return (img_t * 255.0).detach().squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read(file):
    if file.endswith(".float3"):
        return readFloat(file)
    elif file.endswith(".flo"):
        return readFlow(file)
    elif file.endswith(".ppm"):
        return readImage(file)
    elif file.endswith(".pgm"):
        return readImage(file)
    elif file.endswith(".png"):
        return readImage(file)
    elif file.endswith(".jpg"):
        return readImage(file)
    elif file.endswith(".pfm"):
        return readPFM(file)[0]
    else:
        raise Exception("don't know how to read %s" % file)


def write(file, data):
    if file.endswith(".float3"):
        return writeFloat(file, data)
    elif file.endswith(".flo"):
        return writeFlow(file, data)
    elif file.endswith(".ppm"):
        return writeImage(file, data)
    elif file.endswith(".pgm"):
        return writeImage(file, data)
    elif file.endswith(".png"):
        return writeImage(file, data)
    elif file.endswith(".jpg"):
        return writeImage(file, data)
    elif file.endswith(".pfm"):
        return writePFM(file, data)
    else:
        raise Exception("don't know how to write %s" % file)


def readPFM(file):
    file = open(file, "rb")

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == "PF":
        color = True
    elif header.decode("ascii") == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:
        endian = "<"
        scale = -scale
    else:
        endian = ">"

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def writePFM(file, image, scale=1):
    file = open(file, "wb")

    color = None

    if image.dtype.name != "float32":
        raise Exception("Image dtype must be float32.")

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:
        color = False
    else:
        raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

    file.write("PF\n" if color else "Pf\n".encode())
    file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == "<" or endian == "=" and sys.byteorder == "little":
        scale = -scale

    file.write("%f\n".encode() % scale)

    image.tofile(file)


def readFlow(name):
    if name.endswith(".pfm") or name.endswith(".PFM"):
        return readPFM(name)[0][:, :, 0:2]

    f = open(name, "rb")

    header = f.read(4)
    if header.decode("utf-8") != "PIEH":
        raise Exception("Flow file header does not contain PIEH")

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)


def readImage(name):
    if name.endswith(".pfm") or name.endswith(".PFM"):
        data = readPFM(name)[0]
        if len(data.shape) == 3:
            return data[:, :, 0:3]
        else:
            return data
    return imread(name)


def writeImage(name, data):
    if name.endswith(".pfm") or name.endswith(".PFM"):
        return writePFM(name, data, 1)
    return imwrite(name, data)


def writeFlow(name, flow):
    f = open(name, "wb")
    f.write("PIEH".encode("utf-8"))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)


def readFloat(name):
    f = open(name, "rb")

    if (f.readline().decode("utf-8")) != "float\n":
        raise Exception("float file %s did not contain <float> keyword" % name)

    dim = int(f.readline())

    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d

    dims = list(reversed(dims))

    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim > 2:
        data = np.transpose(data, (2, 1, 0))
        data = np.transpose(data, (1, 0, 2))

    return data


def writeFloat(name, data):
    f = open(name, "wb")

    dim = len(data.shape)
    if dim > 3:
        raise Exception("bad float file dimension: %d" % dim)

    f.write(("float\n").encode("ascii"))
    f.write(("%d\n" % dim).encode("ascii"))

    if dim == 1:
        f.write(("%d\n" % data.shape[0]).encode("ascii"))
    else:
        f.write(("%d\n" % data.shape[1]).encode("ascii"))
        f.write(("%d\n" % data.shape[0]).encode("ascii"))
        for i in range(2, dim):
            f.write(("%d\n" % data.shape[i]).encode("ascii"))

    data = data.astype(np.float32)
    if dim == 2:
        data.tofile(f)

    else:
        np.transpose(data, (2, 0, 1)).tofile(f)


def check_dim_and_resize(tensor_list):
    shape_list = []
    for t in tensor_list:
        shape_list.append(t.shape[2:])

    if len(set(shape_list)) > 1:
        desired_shape = shape_list[0]
        print(f"Inconsistent size of input video frames. All frames will be resized to {desired_shape}")

        resize_tensor_list = []
        for t in tensor_list:
            resize_tensor_list.append(torch.nn.functional.interpolate(t, size=tuple(desired_shape), mode="bilinear"))

        tensor_list = resize_tensor_list

    return tensor_list

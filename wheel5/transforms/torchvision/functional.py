from numbers import Number
from typing import Union, Tuple

import numpy as np
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as F


def inv_normalize(tensor: Tensor, mean, std, inplace=False) -> Tensor:
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    inv_mean = -mean / std
    inv_std = 1.0 / std
    return F.normalize(tensor, inv_mean.tolist(), inv_std.tolist(), inplace)


def pad_to_square(img: Image, fill: Union[Number, str, tuple] = 0) -> Image:
    w, h = img.size
    if w != h:
        diff = max(w, h) - min(w, h)
        d1 = int(diff / 2)
        d2 = diff - d1

        padding = (0, d1, 0, d2) if w > h else (d1, 0, d2, 0)
        img = F.pad(img, padding, fill)

    return img


def square_padded_resize(img: Image, size: Union[Tuple[int, int], int], fill: Union[Number, str, tuple] = 0, interpolation: int = Image.BILINEAR) -> Image:
    w, h = img.size
    if w != h:
        diff = max(w, h) - min(w, h)
        d1 = int(diff / 2)
        d2 = diff - d1

        padding = (0, d1, 0, d2) if w > h else (d1, 0, d2, 0)
        img = F.pad(img, padding, fill)

    return F.resize(img, size, interpolation)


def rescale(img: Image, scale: float, interpolation: int = Image.BILINEAR) -> Image:
    w, h = img.size

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    return F.resize(img, (new_h, new_w), interpolation)

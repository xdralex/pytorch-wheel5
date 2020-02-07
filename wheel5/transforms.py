from numbers import Number
from typing import Union, Tuple

import numpy as np
from PIL import Image
from torchvision.transforms import functional as F


class InvNormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)

        inv_mean = -mean / std
        inv_std = 1.0 / std
        return F.normalize(tensor, inv_mean.tolist(), inv_std.tolist(), self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class PadToSquare(object):
    def __init__(self, fill: Union[Number, str, tuple] = 0):
        self.fill = fill

    def __call__(self, img: Image) -> Image:
        w, h = img.size
        if w != h:
            diff = max(w, h) - min(w, h)
            d1 = int(diff / 2)
            d2 = diff - d1

            padding = (0, d1, 0, d2) if w > h else (d1, 0, d2, 0)
            img = F.pad(img, padding, self.fill)

        return img

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(fill={self.fill})'


class SquarePaddedResize(object):
    def __init__(self, size: Union[Tuple[int, int], int], fill: Union[Number, str, tuple] = 0, interpolation: int = Image.BILINEAR):
        self.size = size
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, img: Image) -> Image:
        w, h = img.size
        if w != h:
            diff = max(w, h) - min(w, h)
            d1 = int(diff / 2)
            d2 = diff - d1

            padding = (0, d1, 0, d2) if w > h else (d1, 0, d2, 0)
            img = F.pad(img, padding, self.fill)

        return F.resize(img, self.size, self.interpolation)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size}, fill={self.fill}, interpolation={self.interpolation})'


class Rescale(object):
    def __init__(self, scale: float, interpolation: int = Image.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img: Image) -> Image:
        w, h = img.size

        new_w = int(round(w * self.scale))
        new_h = int(round(h * self.scale))

        return F.resize(img, (new_h, new_w), self.interpolation)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.scale}, interpolation={self.interpolation})'

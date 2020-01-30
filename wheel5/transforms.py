from numbers import Number
from typing import Union, Tuple

from PIL import Image
from torchvision.transforms import functional as F


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

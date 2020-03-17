from numbers import Number
from typing import Union, Tuple

from PIL import Image
from torch import Tensor

from .functional import inv_normalize, pad_to_square, square_padded_resize, rescale


class InvNormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor: Tensor):
        return inv_normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class PadToSquare(object):
    def __init__(self, fill: Union[Number, str, tuple] = 0):
        self.fill = fill

    def __call__(self, img: Image) -> Image:
        return pad_to_square(img, self.fill)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(fill={self.fill})'


class SquarePaddedResize(object):
    def __init__(self, size: Union[Tuple[int, int], int], fill: Union[Number, str, tuple] = 0, interpolation: int = Image.BILINEAR):
        self.size = size
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, img: Image) -> Image:
        return square_padded_resize(img, self.size, self.fill, self.interpolation)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size}, fill={self.fill}, interpolation={self.interpolation})'


class Rescale(object):
    def __init__(self, scale: float, interpolation: int = Image.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img: Image) -> Image:
        return rescale(img, self.scale, self.interpolation)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.scale}, interpolation={self.interpolation})'

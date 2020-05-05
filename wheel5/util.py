from typing import TypeVar, Generic, Optional
from typing import Union

import numpy as np
import torch
from PIL.Image import Image as Img

T = TypeVar('T')


class Closure(Generic[T]):
    def __init__(self):
        self.value: Optional[T] = None


def shape(tensor: Union[Img, torch.Tensor, np.ndarray]) -> str:
    if isinstance(tensor, Img):
        w, h = tensor.size
        prefix = 'image'
        dims = [h, w]
    elif isinstance(tensor, torch.Tensor):
        prefix = 'tensor'
        dims = list(tensor.shape)
    elif isinstance(tensor, np.ndarray):
        prefix = 'ndarray'
        dims = list(tensor.shape)
    else:
        raise NotImplementedError(f'image type: {type(tensor)}')

    dims = [str(dim) for dim in dims]
    return f'{prefix}({"x".join(dims)})'

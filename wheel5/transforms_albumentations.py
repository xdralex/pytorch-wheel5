from numbers import Number
from typing import Union

import cv2
from albumentations import ImageOnlyTransform
from albumentations.augmentations import functional as AF

import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF


class PadToSquare(ImageOnlyTransform):
    def __init__(self,
                 fill: Union[Number, str, tuple] = 0,
                 always_apply=False,
                 p=1.0):
        super(PadToSquare, self).__init__(always_apply, p)

        self.fill = fill

    def apply(self, img, **params):
        h, w = img.shape[:2]

        if w != h:
            diff = max(w, h) - min(w, h)
            d1 = int(diff / 2)
            d2 = diff - d1

            left, top, right, bottom = (0, d1, 0, d2) if w > h else (d1, 0, d2, 0)
            img = AF.pad_with_params(img,
                                     h_pad_top=top,
                                     h_pad_bottom=bottom,
                                     w_pad_left=left,
                                     w_pad_right=right,
                                     border_mode=cv2.BORDER_CONSTANT,
                                     value=self.fill)

        return img

    def get_transform_init_args_names(self):
        return "fill",

    def get_params_dependent_on_targets(self, params):
        raise NotImplementedError()


class Resize(ImageOnlyTransform):
    def __init__(self,
                 height: int,
                 width: int,
                 interpolation: int = Image.BILINEAR,
                 always_apply=False,
                 p=1.0):
        super(Resize, self).__init__(always_apply, p)

        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, img, **params):
        img = Image.fromarray(img)
        img = TF.resize(img, (self.height, self.width), self.interpolation)
        return np.array(img)

    def get_transform_init_args_names(self):
        return "height", "width", "interpolation"

    def get_params_dependent_on_targets(self, params):
        raise NotImplementedError()

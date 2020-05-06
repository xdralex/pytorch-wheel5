from numbers import Number
from typing import Union

import cv2
from albumentations import ImageOnlyTransform, DualTransform

from .functional import rescale, pad_to_square


class Rescale(ImageOnlyTransform):
    def __init__(self, scale: float, interpolation: int = cv2.INTER_AREA, always_apply=False, p=1.0):
        super(Rescale, self).__init__(always_apply, p)

        self.scale = scale
        self.interpolation = interpolation

    def apply(self, img, **params):
        return rescale(img, self.scale, self.interpolation)

    def get_transform_init_args_names(self):
        return "scale", "interpolation"

    def get_params_dependent_on_targets(self, params):
        raise NotImplementedError()


class PadToSquare(ImageOnlyTransform):
    def __init__(self, fill: Union[Number, str, tuple] = 0, always_apply=False, p=1.0):
        super(PadToSquare, self).__init__(always_apply, p)

        self.fill = fill

    def apply(self, img, **params):
        return pad_to_square(img, self.fill)

    def get_transform_init_args_names(self):
        return "fill",

    def get_params_dependent_on_targets(self, params):
        raise NotImplementedError()


class Identity(DualTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(Identity, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return img

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def get_transform_init_args_names(self):
        return ()

    def get_params_dependent_on_targets(self, params):
        raise NotImplementedError()

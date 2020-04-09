from numbers import Number
from typing import Union

from albumentations import ImageOnlyTransform

from .functional import pad_to_square


class PadToSquare(ImageOnlyTransform):
    def __init__(self, fill: Union[Number, str, tuple] = 0,
                 always_apply=False, p=1.0):
        super(PadToSquare, self).__init__(always_apply, p)

        self.fill = fill

    def apply(self, img, **params):
        return pad_to_square(img, self.fill)

    def get_transform_init_args_names(self):
        return "fill",

    def get_params_dependent_on_targets(self, params):
        raise NotImplementedError()

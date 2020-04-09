from numbers import Number
from typing import Union

import cv2
from albumentations.augmentations import functional as AF


def pad_to_square(img, fill: Union[Number, str, tuple] = 0):
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
                                 value=fill)

    return img

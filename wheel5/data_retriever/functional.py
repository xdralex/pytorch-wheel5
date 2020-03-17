from typing import Optional, Tuple

import torch
from numpy.random.mtrand import RandomState
import numpy as np
from torch import Tensor


def mixup(img1: Tensor, lb1: Tensor,
          img2: Tensor, lb2: Tensor,
          alpha: float, random_state: Optional[RandomState] = None) -> Tuple[Tensor, Tensor]:
    # img shape: (N, F, H, W)
    # lb shape: (N, C)

    random_state = random_state or RandomState()
    q = random_state.beta(a=alpha, b=alpha)

    with torch.no_grad():
        assert img1.shape == img2.shape
        assert lb1.shape == lb2.shape
        assert len(img1.shape) == 4
        assert len(lb1.shape) == 2

        img = torch.lerp(img1, img2, weight=q)
        lb = torch.lerp(lb1, lb2, weight=q)

        return img, lb


def cutmix(img1: Tensor, lb1: Tensor,
           img2: Tensor, lb2: Tensor,
           alpha: float, sync: bool, random_state: Optional[RandomState] = None) -> Tuple[Tensor, Tensor]:
    # img shape: (N, F, H, W)
    # lb shape: (N, C)

    random_state = random_state or RandomState()
    q = random_state.beta(a=alpha, b=alpha)

    with torch.no_grad():
        assert img1.shape == img2.shape
        assert lb1.shape == lb2.shape
        assert len(img1.shape) == 4
        assert len(lb1.shape) == 2

        h, w = img1.shape[-2:]

        factor = np.sqrt(1 - q)
        patch_h, patch_w = int(np.round(h * factor)), int(np.round(w * factor))
        sample_h, sample_w = h - patch_h + 1, w - patch_w + 1

        bb1_y1, bb1_x1 = random_state.randint(sample_h), random_state.randint(sample_w)
        bb1_y2, bb1_x2 = bb1_y1 + patch_h, bb1_x1 + patch_w        

        if sync:
            bb2_y1, bb2_x1 = bb1_y1, bb1_x1
            bb2_y2, bb2_x2 = bb1_y2, bb1_x2
        else:
            bb2_y1, bb2_x1 = random_state.randint(sample_h), random_state.randint(sample_w)
            bb2_y2, bb2_x2 = bb2_y1 + patch_h, bb2_x1 + patch_w

        img = img2.clone()
        img[:, :, bb2_y1:bb2_y2, bb2_x1:bb2_x2] = img1[:, :, bb1_y1:bb1_y2, bb1_x1:bb1_x2]

        weight = 1.0 - patch_h * patch_w / float(h * w)
        lb = torch.lerp(lb1, lb2, weight=weight)

        return img, lb

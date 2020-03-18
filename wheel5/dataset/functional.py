from typing import Optional, Tuple, List

import numpy as np
import torch
from numpy.random.mtrand import RandomState
from torch import Tensor


def class_distribution(targets: List[int], classes: int) -> np.ndarray:
    counts = np.zeros(classes)
    for target in targets:
        assert 0 <= target < classes
        counts[target] += 1

    return counts


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


def rand_bbox(w: int, h: int, q: float, random_state: Optional[RandomState] = None) -> Tuple[int, int, int, int]:
    random_state = random_state or RandomState()

    factor = np.sqrt(q)
    patch_w, patch_h = int(np.round(w * factor)), int(np.round(h * factor))

    cx, cy = random_state.randint(w), random_state.randint(h)

    bb_x1 = int(np.clip(cx - patch_w // 2, 0, w))
    bb_y1 = int(np.clip(cy - patch_h // 2, 0, h))
    bb_x2 = int(np.clip(cx + patch_w // 2, 0, w))
    bb_y2 = int(np.clip(cy + patch_h // 2, 0, h))

    return bb_x1, bb_y1, bb_x2, bb_y2


def cutmix(img1: Tensor, lb1: Tensor,
           img2: Tensor, lb2: Tensor,
           alpha: float, mode: str = 'compact', random_state: Optional[RandomState] = None) -> Tuple[Tensor, Tensor]:
    # img shape: (N, F, H, W)
    # lb shape: (N, C)

    assert mode in ['compact', 'halo', 'both']

    random_state = random_state or RandomState()
    q = random_state.beta(a=alpha, b=alpha)

    if (mode == 'compact' and q > 0.5) or (mode == 'halo' and q < 0.5):
        q = 1 - q

    with torch.no_grad():
        assert img1.shape == img2.shape
        assert lb1.shape == lb2.shape
        assert len(img1.shape) == 4
        assert len(lb1.shape) == 2

        h, w = img1.shape[-2:]
        bb_x1, bb_y1, bb_x2, bb_y2 = rand_bbox(h, w, q, random_state)

        img = img2.clone()
        img[:, :, bb_y1:bb_y2, bb_x1:bb_x2] = img1[:, :, bb_y1:bb_y2, bb_x1:bb_x2]

        weight = 1.0 - (bb_y2 - bb_y1) * (bb_x2 - bb_x1) / float(h * w)
        lb = torch.lerp(lb1, lb2, weight=weight)

        return img, lb

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

    counts = counts / float(len(targets))
    return counts


def mixup(img1: Tensor, lb1: Tensor,
          img2: Tensor, lb2: Tensor,
          alpha: float,
          random_state: Optional[RandomState] = None) -> Tuple[Tensor, Tensor]:
    # img shape: (F, H, W)
    # lb shape: (C)

    random_state = random_state or RandomState()
    q = alpha  # random_state.beta(a=alpha, b=alpha)

    with torch.no_grad():
        assert len(img1.shape) == 3 and len(img2.shape) == 3
        assert len(lb1.shape) == 1 and len(lb2.shape) == 1
        assert img1.shape == img2.shape
        assert lb1.shape == lb2.shape

        img = torch.lerp(img1, img2, weight=q)
        lb = torch.lerp(lb1, lb2, weight=q)

        return img, lb


def cutmix(img_src: Tensor, lb_src: Tensor,
           img_dst: Tensor, lb_dst: Tensor,
           alpha: float, mode: str = 'compact',
           random_state: Optional[RandomState] = None) -> Tuple[Tensor, Tensor]:
    # img shape: (F, H, W)
    # lb shape: (C)

    assert mode in ['compact', 'halo', 'both']

    random_state = random_state or RandomState()
    q = random_state.beta(a=alpha, b=alpha)

    if (mode == 'compact' and q > 0.5) or (mode == 'halo' and q < 0.5):
        q = 1 - q

    with torch.no_grad():
        assert len(img_src.shape) == 3 and len(img_dst.shape) == 3
        assert len(lb_src.shape) == 1 and len(lb_dst.shape) == 1
        assert img_src.shape[0] == img_dst.shape[0]
        assert lb_src.shape == lb_dst.shape

        src_h, src_w = img_src.shape[1:]
        dst_h, dst_w = img_dst.shape[1:]

        factor = np.sqrt(q)
        patch_h, patch_w = int(np.round(dst_h * factor)), int(np.round(dst_w * factor))
        patch_h, patch_w = min(patch_h, src_h), min(patch_w, src_w)

        dst_cy = random_state.randint(dst_h)
        dst_cx = random_state.randint(dst_w)

        dst_y1 = int(np.clip(dst_cy - patch_h // 2, 0, dst_h))
        dst_y2 = int(np.clip(dst_cy + patch_h // 2, 0, dst_h))
        dst_x1 = int(np.clip(dst_cx - patch_w // 2, 0, dst_w))
        dst_x2 = int(np.clip(dst_cx + patch_w // 2, 0, dst_w))

        patch_h, patch_w = dst_y2 - dst_y1, dst_x2 - dst_x1

        src_y1 = random_state.randint(src_h - patch_h + 1)
        src_y2 = src_y1 + patch_h
        src_x1 = random_state.randint(src_w - patch_w + 1)
        src_x2 = src_x1 + patch_w

        img = img_dst.clone()
        img[:, dst_y1:dst_y2, dst_x1:dst_x2] = img_src[:, src_y1:src_y2, src_x1:src_x2]

        weight = float(patch_h * patch_w) / float(dst_h * dst_w)
        lb = torch.lerp(lb_dst, lb_src, weight=weight)

        return img, lb

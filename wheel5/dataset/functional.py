from typing import Optional, Tuple, List

import numpy as np
import torch
from numpy.random.mtrand import RandomState

from wheel5.tricks.heatmap import heatmap_to_selection_mask


def mixup(img_src: torch.Tensor, lb_src: torch.Tensor,
          img_dst: torch.Tensor, lb_dst: torch.Tensor,
          alpha: float,
          random_state: Optional[RandomState] = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
    # img shape: (F, H, W)
    # lb shape: (C)

    random_state = random_state or RandomState()
    weight = random_state.beta(a=alpha, b=alpha)

    assert len(img_src.shape) == 3 and len(img_dst.shape) == 3
    assert len(lb_src.shape) == 1 and len(lb_dst.shape) == 1
    assert img_src.shape == img_dst.shape
    assert lb_src.shape == lb_dst.shape

    img = torch.lerp(img_src, img_dst, weight=weight)
    lb = torch.lerp(lb_src, lb_dst, weight=weight)

    return img, lb, weight


def attentive_cutmix(img_src: torch.Tensor, lb_src: torch.Tensor,
                     img_dst: torch.Tensor, lb_dst: torch.Tensor,
                     heatmap_src: torch.Tensor,
                     alpha: float, q_min: float = 0.0, q_max: float = 1.0, mode: str = 'compact',
                     random_state: Optional[RandomState] = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
    # img shape: (F, H, W)
    # lb shape: (C)
    # heatmap shape: (H, W)

    random_state = random_state or RandomState()
    q = random_state.beta(a=alpha, b=alpha)

    if (mode == 'compact' and q > 0.5) or (mode == 'halo' and q < 0.5):
        q = 1 - q

    q = min(max(q, q_min), q_max)

    heatmap_expanded = heatmap_src.unsqueeze(dim=0)
    mask_src = heatmap_to_selection_mask(heatmap_expanded, q)
    mask_src = mask_src.squeeze(dim=0)

    assert len(img_src.shape) == 3 and len(img_dst.shape) == 3
    assert len(lb_src.shape) == 1 and len(lb_dst.shape) == 1
    assert len(mask_src.shape) == 2

    assert img_src.shape == img_dst.shape
    assert lb_src.shape == lb_dst.shape

    assert img_src.shape[1:] == mask_src.shape

    img = img_dst.new_zeros(img_dst.shape)
    img[:, mask_src] = img_src[:, mask_src]
    img[:, ~mask_src] = img_dst[:, ~mask_src]

    weight = float(mask_src.int().sum()) / float(mask_src.numel())
    lb = torch.lerp(lb_dst, lb_src, weight)

    return img, lb, weight


def cutmix(img_src: torch.Tensor, lb_src: torch.Tensor,
           img_dst: torch.Tensor, lb_dst: torch.Tensor,
           alpha: float, q_min: float = 0.0, q_max: float = 1.0, mode: str = 'compact',
           random_state: Optional[RandomState] = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
    # img shape: (F, H, W)
    # lb shape: (C)

    assert mode in ['compact', 'halo', 'both']

    random_state = random_state or RandomState()
    q = random_state.beta(a=alpha, b=alpha)

    if (mode == 'compact' and q > 0.5) or (mode == 'halo' and q < 0.5):
        q = 1 - q

    q = min(max(q, q_min), q_max)

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

    return img, lb, weight

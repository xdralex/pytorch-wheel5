import math

import torch
import torch.nn.functional as F


def heatmap_to_selection_mask(heatmap: torch.Tensor, cutoff_ratio: float) -> torch.Tensor:
    # heatmap shape: (N, U, V)
    n, u, v = heatmap.shape
    assert 0 <= cutoff_ratio <= 1

    heatmap_sorted, _ = heatmap.view(n, -1).sort(dim=1, descending=True)
    _, cells = heatmap_sorted.shape

    limit = max(min(int(round(cells * cutoff_ratio)), cells), 0)
    if limit == 0:
        cutoff = math.inf
    else:
        cutoff, _ = heatmap_sorted[:, 0:limit].min(dim=1)
        cutoff = cutoff.view(n, 1, 1)

    return torch.ge(heatmap, cutoff)


def upsample_heatmap(heatmap: torch.Tensor, h: int, w: int, inter_mode: str) -> torch.Tensor:
    # saliency_map - shape: (N, U, V)
    n, _, _ = heatmap.shape

    align_corners = False if inter_mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None

    heatmap = heatmap.unsqueeze(dim=1)                      # shape: (N, 1, U, V)
    heatmap = F.interpolate(heatmap,                        # shape: (N, 1, H, W)
                            size=(h, w),
                            mode=inter_mode,
                            align_corners=align_corners)
    heatmap = heatmap.squeeze(dim=1)                        # shape: (N, H, W)

    return heatmap


def normalize_heatmap(heatmap: torch.Tensor):
    # saliency_map - shape: (N, H, W)
    n, _, _ = heatmap.shape

    heatmap_max, _ = heatmap.view(n, -1).max(dim=1)
    heatmap_min, _ = heatmap.view(n, -1).min(dim=1)

    heatmap_max = heatmap_max.view(n, 1, 1)
    heatmap_min = heatmap_min.view(n, 1, 1)

    return (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

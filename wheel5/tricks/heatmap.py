import math

import torch
import torch.nn.functional as F


def heatmap_to_selection_mask(heatmap: torch.Tensor, h: int, w: int, cells: int, inter_mode: str) -> torch.Tensor:
    # heatmaps shape: (N, U, V)
    n, u, v = heatmap.shape
    assert 0 <= cells < u * v

    if cells == 0:
        cutoff = math.inf
    else:
        heatmap_sorted, _ = heatmap.view(n, -1).sort(dim=1, descending=True)
        cutoff, _ = heatmap_sorted[:, 0:cells].min(dim=1)
        cutoff = cutoff.view(n, 1, 1)

    heatmap = upsample_heatmap(heatmap, h, w, inter_mode)
    return torch.ge(heatmap, cutoff).int()


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

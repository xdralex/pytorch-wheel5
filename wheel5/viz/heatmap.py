from dataclasses import dataclass
from typing import List

from abc import ABC, abstractmethod
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import functional as VTF


class HeatmapMode(ABC):
    @abstractmethod
    def combine(self, image: torch.Tensor, mask: torch.Tensor):
        pass


@dataclass
class HeatmapEntry:
    name: str
    y_class: torch.Tensor
    mask: torch.Tensor
    mode: HeatmapMode


class HeatmapModeColormap(HeatmapMode):
    def __init__(self, alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET):
        self.alpha = alpha
        self.colormap = colormap

    def combine(self, image: torch.Tensor, mask: torch.Tensor):
        # image - shape: (3, H, W)
        # mask - shape: (H, W)

        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)     # shape: (H, W, 3), BGR
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float() / 255      # shape: (3, H, W), BGR
        heatmap = heatmap[[2, 1, 0]]                                            # shape: (3, H, W), RGB

        return image * self.alpha + heatmap * (1 - self.alpha)


class HeatmapModeBloom(HeatmapMode):
    def combine(self, image: torch.Tensor, mask: torch.Tensor):
        # image - shape: (3, H, W)
        # mask - shape: (H, W)

        return torch.stack([image[i] * mask for i in range(0, image.shape[0])])


def draw_heatmap(x: torch.Tensor,
                 entries: List[HeatmapEntry],
                 target_classes: List[str],
                 orientation: str = 'vertical',
                 width: float = 4,
                 height: float = 4,
                 fontsize: int = 9):

    # x - shape: (N, 3, H, W)
    # y_class - shape: (N)
    # mask - shape: (N, H, W)

    assert len(x.shape) == 4
    assert x.shape[1] == 3

    for entry in entries:
        assert len(entry.y_class.shape) == 1
        assert len(entry.mask.shape) == 3

        assert x.shape[0] == entry.y_class.shape[0]
        assert x.shape[0] == entry.mask.shape[0]

        assert x.shape[2:] == entry.mask.shape[1:]

    n, _, h, w = x.shape

    if orientation == 'vertical':
        cols, rows = len(entries), n
    elif orientation == 'horizontal':
        cols, rows = n, len(entries)
    else:
        raise AssertionError(f'Invalid orientation: {orientation}')

    fig = plt.figure(figsize=(width * cols, height * rows))

    for i in range(0, n):
        for j in range(0, len(entries)):
            entry = entries[j]

            plt.subplot(rows, cols, len(entries) * i + j + 1)
            plt.title(f'{entry.name}: {target_classes[int(entry.y_class[i])]}', fontsize=fontsize)
            plt.grid(False)
            plt.axis('off')

            image = entry.mode.combine(x[i], entry.mask[i])
            plt.imshow(VTF.to_pil_image(image))

    return fig

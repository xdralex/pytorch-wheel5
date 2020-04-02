import torch
from torchvision.transforms import functional as VTF

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from numpy.random.mtrand import RandomState
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from typing import List
import cv2


def draw_heatmap(x: torch.Tensor,
                 y_class: torch.Tensor,
                 y_class_hat: torch.Tensor,
                 mask: torch.Tensor,
                 mask_hat: torch.Tensor,
                 target_classes: List[str],
                 fontsize: int = 9,
                 alpha: float = 0.5):

    def convert_mask(m: torch.Tensor):
        hm = cv2.applyColorMap(np.uint8(255 * m), cv2.COLORMAP_JET)     # shape: (H, W, 3), BGR
        hm = torch.from_numpy(hm).permute(2, 0, 1).float() / 255        # shape: (3, H, W), BGR
        hm = hm[[2, 1, 0]]                                              # shape: (3, H, W), RGB
        return hm

    # x - shape: (N, 3, H, W)
    # y_class - shape: (N)
    # y_class_hat - shape: (N)
    # mask - shape: (N, H, W)
    # mask_hat - shape: (N, H, W)

    assert len(x.shape) == 4
    assert len(y_class.shape) == 1
    assert len(y_class_hat.shape) == 1
    assert len(mask.shape) == 3
    assert len(mask_hat.shape) == 3

    assert x.shape[1] == 3
    assert x.shape[2:] == mask.shape[1:]
    assert x.shape[2:] == mask_hat.shape[1:]

    for tensor in [y_class, y_class_hat, mask, mask_hat]:
        assert x.shape[0] == tensor.shape[0]

    n, _, h, w = x.shape

    fig = plt.figure(figsize=(5*2, 5*n))

    for i in range(0, n):
        heatmap = convert_mask(mask[i])
        heatmap_hat = convert_mask(mask_hat[i])

        plt.subplot(n, 2, 2*i + 1)
        plt.imshow(VTF.to_pil_image(x[i]))
        plt.imshow(VTF.to_pil_image(heatmap), alpha=alpha)
        plt.title(f'actual: {target_classes[int(y_class[i])]}', fontsize=fontsize)
        plt.grid(False)
        plt.axis('off')

        plt.subplot(n, 2, 2*i + 2)
        plt.imshow(VTF.to_pil_image(x[i]))
        plt.imshow(VTF.to_pil_image(heatmap_hat), alpha=alpha)
        plt.title(f'predicted: {target_classes[int(y_class_hat[i])]}', fontsize=fontsize)
        plt.grid(False)
        plt.axis('off')

    return fig

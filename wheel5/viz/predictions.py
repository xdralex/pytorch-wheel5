from typing import List

import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pandas import np
from torchvision.transforms import functional as VTF

from wheel5.tasks.detection import BoundingBoxes


def draw_classes(x: torch.Tensor,
                 y_probs_hat: torch.Tensor,
                 target_classes: List[str],
                 top_n: int = 5,
                 orientation: str = 'vertical',
                 width: float = 5,
                 height: float = 5,
                 fontsize: int = 9):
    # x - shape: (N, 3, H, W)
    # y_probs_hat - shape: (N, C)

    assert len(x.shape) == 4
    assert x.shape[1] == 3

    assert len(y_probs_hat.shape) == 2
    assert x.shape[0] == y_probs_hat.shape[0]
    assert y_probs_hat.shape[1] == len(target_classes)

    n, _, h, w = x.shape

    if orientation == 'vertical':
        cols, rows = 1, n
    elif orientation == 'horizontal':
        cols, rows = n, 1
    else:
        raise AssertionError(f'Invalid orientation: {orientation}')

    fig = plt.figure(figsize=(width * cols, height * rows))

    for i in range(0, n):
        plt.subplot(rows, cols, i + 1)

        image = x[i].cpu()
        image = VTF.to_pil_image(image)

        probs = y_probs_hat[i].cpu().numpy()
        top_indices = np.argsort(probs)[::-1][:top_n]

        title = ', '.join([f'{target_classes[idx]}={probs[idx]:.3f}' for idx in top_indices])
        plt.title(title, fontsize=fontsize)
        plt.grid(False)
        plt.axis('off')
        plt.imshow(image)

    return fig


def draw_bboxes(x: List[torch.Tensor],
                meta: List[BoundingBoxes],
                classes: List[str],
                orientation: str = 'vertical',
                width: float = 15,
                height: float = 15,
                fontsize: int = 9):

    assert len(x) == len(meta)
    n = len(x)

    if orientation == 'vertical':
        cols, rows = 1, n
    elif orientation == 'horizontal':
        cols, rows = n, 1
    else:
        raise AssertionError(f'Invalid orientation: {orientation}')

    fig = plt.figure(figsize=(width * cols, height * rows))

    for i in range(0, n):
        plt.subplot(rows, cols, i + 1)

        image = x[i].cpu()
        image = VTF.to_pil_image(image)

        image_arr = np.asarray(image)
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)

        for j in range(0, len(meta[i])):
            bbox = meta[i].bboxes[j].cpu().int().numpy()
            label = meta[i].labels[j].cpu().numpy()
            score = meta[i].scores[j].cpu().numpy()

            print(f'bbox={bbox}, label={classes[label]}')

            image_arr = cv2.rectangle(image_arr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 4)

        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_arr)

        plt.grid(False)
        plt.axis('off')
        plt.imshow(image)

    return fig

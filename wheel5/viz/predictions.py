from typing import List

import matplotlib.pyplot as plt
import torch
from pandas import np
from torchvision.transforms import functional as VTF


def draw_predictions(x: torch.Tensor,
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
        probs = y_probs_hat[i].cpu().numpy()
        top_indices = np.argsort(probs)[::-1][:top_n]

        title = ', '.join([f'{target_classes[idx]}={probs[idx]:.3f}' for idx in top_indices])
        plt.title(title, fontsize=fontsize)
        plt.grid(False)
        plt.axis('off')

        plt.imshow(VTF.to_pil_image(image))

    return fig

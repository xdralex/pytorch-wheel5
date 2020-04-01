import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.random.mtrand import RandomState
from torch.utils.data import Dataset


def draw_samples(dataset: Dataset,
                 cols: int, rows: int,
                 width: float = 3, height: float = 3, fontsize: int = 8,
                 random_state: RandomState = None) -> Figure:
    random_state = random_state or np.random.RandomState()

    count = cols * rows
    indices = random_state.choice(np.arange(len(dataset)), count, replace=False)

    fig = plt.figure(figsize=(cols * width, rows * height))

    for i, index in enumerate(indices):
        image, cls, name = dataset[index]

        plt.subplot(rows, cols, i + 1)
        plt.title(f'#{index} - [{cls}] {name}', fontsize=fontsize)
        plt.imshow(image)
        plt.grid(False)
        plt.axis('off')

    return fig

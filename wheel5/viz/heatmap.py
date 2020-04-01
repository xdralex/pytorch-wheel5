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


def draw_heatmap(mask: torch.Tensor, img: torch.Tensor):
    fig = plt.figure(figsize=(5, 5))

    img = VTF.to_pil_image(img)

    plt.imshow(img)
    plt.grid(False)
    plt.axis('off')

    return fig

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from numpy.random.mtrand import RandomState
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from typing import List


def draw_confusion_matrix(classes: List[str], y_true: np.ndarray, y_pred: np.ndarray,
                          cellsize: float = 0.6, fontsize: int = 7) -> Figure:
    n = len(classes)

    cm = confusion_matrix(y_true, y_pred, normalize='all')
    assert cm.shape == (n, n)

    fig = plt.figure(figsize=(n * cellsize, n * cellsize))

    ax = sns.heatmap(cm,
                     cmap='coolwarm',
                     square=True,
                     cbar=False,
                     annot=True,
                     fmt='.2%',
                     annot_kws={'size': fontsize})

    ax.set_xlabel('predicted')
    ax.set_ylabel('actual')

    ax.set_xticklabels(classes, fontsize=fontsize, rotation=45, ha='right')
    ax.set_yticklabels(classes, fontsize=fontsize, rotation=45)

    return fig


def draw_top_errors(classes: List[str], y_true: np.ndarray, y_pred: np.ndarray,
                    image_indices: np.ndarray, image_dataset: Dataset,
                    top: int = 5, examples: int = 5,
                    width: float = 3, height: float = 3, fontsize: int = 8, titlesize: int = 12,
                    random_state: RandomState = None) -> List[Figure]:
    random_state = random_state or np.random.RandomState()

    n = len(classes)

    cm = confusion_matrix(y_true, y_pred, normalize='all')
    assert cm.shape == (n, n)

    pred_v, true_v = np.meshgrid(np.arange(n), np.arange(n))

    df_cm = pd.DataFrame({'x': np.ravel(cm), 'pred_v': np.ravel(pred_v), 'true_v': np.ravel(true_v)})
    df_cm = df_cm.sort_values(by='x', ascending=False)
    df_cm = df_cm[df_cm['pred_v'] != df_cm['true_v']]
    df_cm = df_cm.head(top)

    df_samples = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'image_index': image_indices})

    figs = []
    for row in df_cm.itertuples():
        value, pred_v, true_v = row.x, row.pred_v, row.true_v

        fig = _draw_error_row(df_samples, value, true_v, pred_v,
                              classes, image_dataset, examples,
                              width, height, fontsize, titlesize,
                              random_state)
        figs.append(fig)

    return figs


def draw_errors(classes: List[str], y_true: np.ndarray, y_pred: np.ndarray, actual_class: str, predicted_class: str,
                image_indices: np.ndarray, image_dataset: Dataset,
                examples: int = 4,
                width: float = 3, height: float = 3, fontsize: int = 8, titlesize: int = 12,
                random_state: RandomState = None) -> Figure:
    random_state = random_state or np.random.RandomState()

    n = len(classes)

    cm = confusion_matrix(y_true, y_pred, normalize='all')
    assert cm.shape == (n, n)

    true_v = classes.index(actual_class)
    pred_v = classes.index(predicted_class)

    value = cm[true_v][pred_v]

    df_samples = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'image_index': image_indices})
    return _draw_error_row(df_samples, value, true_v, pred_v,
                           classes, image_dataset, examples,
                           width, height, fontsize, titlesize,
                           random_state)


def _draw_error_row(df_samples: pd.DataFrame, value: float, true_v: int, pred_v: int,
                    classes: List[str], image_dataset: Dataset, examples: int,
                    width: float, height: float, fontsize: int, titlesize: int,
                    random_state: RandomState) -> Figure:
    df_samples_filtered = df_samples[(df_samples['y_true'] == true_v) & (df_samples['y_pred'] == pred_v)]
    df_samples_filtered = df_samples_filtered.sample(frac=1, random_state=random_state)
    filtered_indices = df_samples_filtered.head(examples)['image_index'].tolist()

    fig = plt.figure(figsize=(examples * width, height))
    fig.suptitle(f'{value:.2%} - actual: {classes[true_v]}, predicted: {classes[pred_v]}', fontsize=titlesize)

    for counter, index in enumerate(filtered_indices):
        image, target, _ = image_dataset[index]

        plt.subplot(1, examples, counter + 1)
        plt.title(f'#{index} - [{target}] {classes[target]}', fontsize=fontsize)
        plt.imshow(image)
        plt.grid(False)
        plt.axis('off')

    return fig

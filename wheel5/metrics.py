from abc import abstractmethod, ABC
from typing import Tuple

import torch
from torch import Tensor
from torch import nn


def exact_match_accuracy(y: Tensor, z: Tensor, y_probs: Tensor, y_hat: Tensor) -> Tuple[float, float]:
    assert y.ndim == 1
    assert y_hat.ndim == 1
    assert y.shape == y_hat.shape

    correct = float(torch.sum(y_hat == y))
    total = float(y.shape[0])

    return correct, total


def jaccard_accuracy(y: Tensor, z: Tensor, y_probs: Tensor, y_hat: Tensor) -> Tuple[float, float]:
    assert y.ndim == 2
    assert y_probs.ndim == 2
    assert y.shape == y_probs.shape

    correct = 0.0
    total = 0.0
    for i in range(0, y.shape[0]):
        intersection = torch.min(y[i], y_probs[i])
        union = torch.max(y[i], y_probs[i])

        correct += float(torch.sum(intersection))
        total += float(torch.sum(union))

    return correct, total


class Accuracy(nn.Module, ABC):
    def __init__(self):
        super(Accuracy, self).__init__()

    @abstractmethod
    def forward(self, y: Tensor, z: Tensor, y_probs: Tensor, y_hat: Tensor) -> Tuple[float, float]:
        pass


class ExactMatchAccuracy(Accuracy):
    def __init__(self):
        super(ExactMatchAccuracy, self).__init__()

    def forward(self, y: Tensor, z: Tensor, y_probs: Tensor, y_hat: Tensor) -> Tuple[float, float]:
        return exact_match_accuracy(y, z, y_probs, y_hat)


class JaccardAccuracy(Accuracy):
    def __init__(self):
        super(JaccardAccuracy, self).__init__()

    def forward(self, y: Tensor, z: Tensor, y_probs: Tensor, y_hat: Tensor) -> Tuple[float, float]:
        return jaccard_accuracy(y, z, y_probs, y_hat)

from abc import abstractmethod, ABC
from typing import Tuple

from torch import Tensor
from torch import nn

from .functional import exact_match_accuracy, jaccard_accuracy


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
        return exact_match_accuracy(y, y_hat)


class JaccardAccuracy(Accuracy):
    def __init__(self):
        super(JaccardAccuracy, self).__init__()

    def forward(self, y: Tensor, z: Tensor, y_probs: Tensor, y_hat: Tensor) -> Tuple[float, float]:
        return jaccard_accuracy(y, y_probs)

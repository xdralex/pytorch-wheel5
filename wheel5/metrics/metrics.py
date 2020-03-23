from typing import Tuple

from torch import Tensor
from torch import nn

from .functional import exact_match_accuracy, jaccard_accuracy


class ExactMatchAccuracy(nn.Module):
    def __init__(self):
        super(ExactMatchAccuracy, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        return exact_match_accuracy(input, target)


class JaccardAccuracy(nn.Module):
    def __init__(self):
        super(JaccardAccuracy, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        return jaccard_accuracy(input, target)

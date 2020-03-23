from typing import Tuple

import torch
from torch import Tensor


def exact_match_accuracy(input: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    assert target.ndim == 1
    assert input.ndim == 1
    assert target.shape == input.shape

    correct = torch.sum(input == target)
    total = input.new_tensor(target.shape[0])

    return correct, total


def jaccard_accuracy(input: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    assert target.ndim == 2
    assert input.ndim == 2
    assert target.shape == input.shape

    correct = input.new_zeros(1)
    total = input.new_zeros(1)
    for i in range(0, target.shape[0]):
        intersection = torch.min(target[i], input[i])
        union = torch.max(target[i], input[i])

        correct += torch.sum(intersection)
        total += torch.sum(union)

    return correct, total

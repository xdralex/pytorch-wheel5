from logging import Logger
from typing import Tuple, Optional

import torch
from torch import Tensor


def exact_match_accuracy(input: Tensor, target: Tensor,
                         name: str = '', logger: Optional[Logger] = None, debug: bool = False) -> Tuple[Tensor, Tensor]:
    assert target.ndim == 1
    assert input.ndim == 1
    assert target.shape == input.shape

    numer = torch.sum(input == target)
    denom = input.new_tensor(target.shape[0])

    if debug:
        out = f'exact match accuracy[{name}]:\n' + \
              f'input={input.shape}, target={target.shape}, ratio={numer}/{denom}\n\n' + \
              f'input: \n{input}\n\n' + \
              f'target: \n{target}\n\n'

        logger.debug(out)

    return numer, denom


def jaccard_accuracy(input: Tensor, target: Tensor,
                     name: str = '', logger: Optional[Logger] = None, debug: bool = False) -> Tuple[Tensor, Tensor]:
    assert target.ndim == 2
    assert input.ndim == 2
    assert target.shape == input.shape

    numer = input.new_zeros(1)
    denom = input.new_zeros(1)
    for i in range(0, target.shape[0]):
        intersection = torch.min(target[i], input[i])
        union = torch.max(target[i], input[i])

        numer += torch.sum(intersection)
        denom += torch.sum(union)

    if debug:
        out = f'jaccard accuracy[{name}]:\n' + \
              f'input={input.shape}, target={target.shape}, ratio={numer}/{denom}\n\n' + \
              f'input: \n{input}\n\n' + \
              f'target: \n{target}\n\n'

        logger.debug(out)

    return numer, denom


def dice_accuracy(input: Tensor, target: Tensor,
                  name: str = '', logger: Optional[Logger] = None, debug: bool = False) -> Tuple[Tensor, Tensor]:
    assert target.ndim == 2
    assert input.ndim == 2
    assert target.shape == input.shape

    numer = input.new_zeros(1)
    denom = input.new_zeros(1)
    for i in range(0, target.shape[0]):
        intersection = torch.min(target[i], input[i])

        numer += 2 * torch.sum(intersection)
        denom += (torch.sum(target[i]) + torch.sum(input[i]))

    if debug:
        out = f'dice accuracy[{name}]:\n' + \
              f'input={input.shape}, target={target.shape}, ratio={numer}/{denom}\n\n' + \
              f'input: \n{input}\n\n' + \
              f'target: \n{target}\n\n'

        logger.debug(out)

    return numer, denom

import logging
from typing import Tuple

from torch import Tensor
from torch import nn

from .functional import exact_match_accuracy, jaccard_accuracy, dice_accuracy


class ExactMatchAccuracy(nn.Module):
    def __init__(self):
        super(ExactMatchAccuracy, self).__init__()

        self.logger = logging.getLogger(f'{__name__}')
        self.debug = self.logger.isEnabledFor(logging.DEBUG)

    def forward(self, input: Tensor, target: Tensor, name: str = '') -> Tuple[Tensor, Tensor]:
        return exact_match_accuracy(input, target, name=name, logger=self.logger, debug=self.debug)


class JaccardAccuracy(nn.Module):
    def __init__(self):
        super(JaccardAccuracy, self).__init__()

        self.logger = logging.getLogger(f'{__name__}')
        self.debug = self.logger.isEnabledFor(logging.DEBUG)

    def forward(self, input: Tensor, target: Tensor, name: str = '') -> Tuple[Tensor, Tensor]:
        return jaccard_accuracy(input, target, name=name, logger=self.logger, debug=self.debug)


class DiceAccuracy(nn.Module):
    def __init__(self):
        super(DiceAccuracy, self).__init__()

        self.logger = logging.getLogger(f'{__name__}')
        self.debug = self.logger.isEnabledFor(logging.DEBUG)

    def forward(self, input: Tensor, target: Tensor, name: str = '') -> Tuple[Tensor, Tensor]:
        return dice_accuracy(input, target, name=name, logger=self.logger, debug=self.debug)

from typing import Optional

from torch import Tensor
from torch.nn import CrossEntropyLoss

from .functional import soft_label_cross_entropy


class SoftLabelCrossEntropyLoss(CrossEntropyLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction', 'smooth_factor', 'smooth_dist']

    def __init__(self,
                 smooth_factor: float,
                 smooth_dist: Tensor,
                 weight: Optional[Tensor] = None,
                 ignore_index: Optional[int] = -100,
                 reduction: Optional[str] = 'mean'):
        super(SoftLabelCrossEntropyLoss, self).__init__(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.smooth_factor = smooth_factor
        self.smooth_dist = smooth_dist

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return soft_label_cross_entropy(input, target,
                                        smooth_factor=self.smooth_factor,
                                        smooth_dist=self.smooth_dist,
                                        weight=self.weight,
                                        ignore_index=self.ignore_index,
                                        reduction=self.reduction)

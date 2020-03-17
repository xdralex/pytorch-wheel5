from typing import Optional, Tuple

import torch
from numpy.random.mtrand import RandomState
from torch import Tensor


def mixup(x1: Tensor, y1: Tensor,
          x2: Tensor, y2: Tensor,
          alpha: float, random_state: Optional[RandomState] = None) -> Tuple[Tensor, Tensor]:

    random_state = random_state or RandomState()

    with torch.no_grad():
        assert x1.shape == x2.shape
        assert y1.shape == y2.shape

        # mixing inputs and targets
        q = random_state.beta(a=alpha, b=alpha)
        x = torch.lerp(x1, x2, weight=q)
        y = torch.lerp(y1, y2, weight=q)

        return x, y

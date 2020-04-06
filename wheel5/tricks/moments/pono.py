from typing import Tuple

import torch


def pono(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    #  x - shape: (N, F, H, W)
    assert x.shape == 4

    mean = x.mean(dim=1)
    std = (x.var(dim=1) + eps).sqrt()

    return mean, std

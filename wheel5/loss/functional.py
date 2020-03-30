from typing import Optional

from torch import Tensor
from torch.nn import functional as F


def soft_label_cross_entropy(input: Tensor,
                             target: Tensor,
                             smooth_factor: float,
                             smooth_dist: Tensor,
                             weight: Optional[Tensor] = None,
                             ignore_index: Optional[int] = -100,
                             reduction: Optional[str] = 'mean') -> Tensor:

    # input  - (q) shape: (N, C, d_1, d_2, ..., d_K)
    # target - (p) shape: (N, C, d_1, d_2, ..., d_K)

    assert input.shape == target.shape

    num_classes = input.shape[1]
    assert smooth_dist.ndim == 1
    assert smooth_dist.shape[0] == num_classes

    order = list(range(0, input.ndim))
    order[1], order[-1] = order[-1], order[1]

    # transforming p shape to (N, d_1, d_2, ..., d_K, C)
    p = target.permute(order)                   # p shape: (N, d_1, d_2, ..., d_K, C)

    smooth_dist = smooth_dist.to(input.device)

    # computing softmax probabilities
    log_q = F.log_softmax(input, dim=1)         # log_q shape: (N, C, d_1, d_2, ..., d_K)
    log_q = log_q.permute(order)                # log_q shape: (N, d_1, d_2, ..., d_K, C)

    assert log_q.shape == p.shape

    # applying weight rescaling
    if weight is not None:
        assert weight.ndim == 1
        assert weight.shape[0] == num_classes
        log_q = log_q * weight

    # mixing target distribution with smoothing distribution
    p = p.type_as(input)
    p.lerp_(smooth_dist, smooth_factor)

    # computing loss and applying ignore index mask
    loss = -(p * log_q).sum(-1)                 # loss shape: (N, d_1, d_2, ..., d_K)
    mask = None
    if ignore_index >= 0:
        mask = target.eq(ignore_index)          # mask shape: (N, d_1, d_2, ..., d_K)
        loss.masked_fill_(mask, 0)

    if reduction == 'mean':
        if mask is not None:
            return loss.sum() / (loss.numel() - int(mask.sum()))
        else:
            return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f'Unsupported reduction "{reduction}"')

import torch
from typing import Callable, Optional


def logit_to_score(class_index: Optional[int] = None) -> Callable[..., torch.Tensor]:
    # logit shape: (N, C)

    def fn(logit: torch.Tensor):
        n, c = logit.shape
        assert 0 <= class_index < n

        if class_index is None:
            values, _ = logit.topk(1, dim=1, sorted=False)
            score = values.squeeze(dim=1)
        else:
            score = logit[:, class_index]

        assert score.shape == 1
        assert score.shape[0] == n

        return score

    return fn

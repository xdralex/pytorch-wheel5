import torch
from typing import Callable, Optional


def logit_to_score(class_index: Optional[int] = None) -> Callable[..., torch.Tensor]:
    def fn(logit: torch.Tensor):
        # logit shape: (1, C)

        assert len(logit.shape) == 2
        assert logit.shape[0] == 1

        if class_index is None:
            values, _ = logit.topk(1, dim=1, sorted=False)
            score = values.squeeze(dim=1)
        else:
            _, c = logit.shape
            assert 0 <= class_index < c

            score = logit[:, class_index]

        return score

    return fn

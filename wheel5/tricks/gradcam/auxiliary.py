import torch
from typing import Callable, Optional


def logit_to_score(class_index: Optional[torch.Tensor] = None) -> Callable[..., torch.Tensor]:
    def fn(logit: torch.Tensor):
        # logit shape: (N, C)
        assert len(logit.shape) == 2

        if class_index is None:
            score, _ = logit.max(dim=1)
        else:
            score = logit.gather(dim=1, index=class_index.unsqueeze(dim=1))
            score = score.squeeze(dim=1)

        return score

    return fn

import torch


def moex(x: torch.Tensor, perm: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    #  x - shape: (N, F, H, W)
    assert len(x.shape) == 4

    mean = x.mean(dim=1, keepdim=True)
    std = (x.var(dim=1, keepdim=True) + eps).sqrt()

    scale = std[perm] / std
    shift = mean[perm] - mean * scale

    return x * scale + shift

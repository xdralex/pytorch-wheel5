import torch


def init_softmax_logits(tensor: torch.Tensor, probs: torch.Tensor):
    with torch.no_grad():
        logits = torch.log(probs)
        tensor.copy_(logits)

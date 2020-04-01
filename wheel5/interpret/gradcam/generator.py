import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
import torch.nn.functional as F
from typing import Optional, Callable, Tuple

from wheel5.util import Closure


class GradCAM(object):
    def __init__(self, model: nn.Module, layer: nn.Module, score_fn: Callable[..., torch.Tensor]):
        self.model = model
        self.layer = layer
        self.score_fn = score_fn

        self.activations_closure = Closure[torch.Tensor]()
        self.gradients_closure = Closure[torch.Tensor]()

        self.forward_handle: Optional[RemovableHandle] = None
        self.backward_handle: Optional[RemovableHandle] = None

        self.entered = False

    def init(self):
        assert not self.entered

        def activations_collector(_, input, output):
            self.activations_closure.value = output

        def gradients_collector(_, input, output):
            self.gradients_closure.value = output

        self.forward_handle = self.layer.register_forward_hook(activations_collector)
        self.backward_handle = self.layer.register_backward_hook(gradients_collector)

    def generate(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input - shape: (N, F, H, W)

        with torch.enable_grad():
            n, f, h, w = input.shape

            output = self.model(input)
            score = self.score_fn(output)                                   # dy_c - shape: (N)

            assert score.shape == 1
            assert score.shape[0] == n

            self.model.zero_grad()
            score.backward()

            activations: torch.Tensor = self.activations_closure.value      # A - shape: (N, K, U, V)
            gradients: torch.Tensor = self.gradients_closure.value          # dy_c / dA - shape: (N, K, U, V)

            _, k, u, v = gradients.shape

            alpha = gradients.view(n, k, -1).mean(dim=2)                    # averaging over UxV for each A_k - shape: (N, K)
            weights = alpha.view(n, k, 1, 1)                                # shape: (N, K, 1, 1)
            lin_comb = (weights * activations).sum(dim=1)                   # shape: (N, U, V)
            saliency_map = F.relu(lin_comb)                                 # shape: (N, U, V)

            saliency_map = F.upsample(saliency_map,                         # shape: (N, H, W)
                                      size=(h, w),
                                      mode='bilinear',
                                      align_corners=False)

            saliency_min = saliency_map.min()
            saliency_max = saliency_map.max()

            saliency_map = (saliency_map - saliency_min) / (saliency_max - saliency_min)
            return saliency_map, output, score


    def close(self):
        assert self.entered

        self.forward_handle.remove()
        self.backward_handle.remove()

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
import torch.nn.functional as F
from typing import Optional, Callable, Tuple

from wheel5.util import Closure


# Inspired by https://github.com/1Konny/gradcam_plus_plus-pytorch
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
            return None

        def gradients_collector(_, input, output):
            self.gradients_closure.value, = output
            return None

        self.forward_handle = self.layer.register_forward_hook(activations_collector)
        self.backward_handle = self.layer.register_backward_hook(gradients_collector)

        self.entered = True

    def generate(self, input: torch.Tensor, inter_mode: str = 'bilinear') -> torch.Tensor:
        # input - shape: (1, F, H, W)

        assert len(input.shape) == 4
        assert input.shape[0] == 1

        with torch.enable_grad():
            n, f, h, w = input.shape

            output: torch.Tensor = self.model(input)                        # y - shape: (1, C)
            assert len(output.shape) == 2
            assert output.shape[0] == 1

            score = self.score_fn(output)                                   # dy_c - scalar

            assert len(score.shape) == 1
            assert score.shape[0] == n

            self.model.zero_grad()
            score.backward()

            activations: torch.Tensor = self.activations_closure.value      # A - shape: (1, K, U, V)
            gradients: torch.Tensor = self.gradients_closure.value          # dy^c / dA - shape: (1, K, U, V)

        with torch.no_grad():
            _, k, u, v = gradients.shape

            alpha = gradients.view(n, k, -1).mean(dim=2)                    # avg over UxV for each A_k - shape: (1, K)
            weights = alpha.view(n, k, 1, 1)                                # α^c_k - shape: (1, K, 1, 1)
            lin_comb = (weights * activations).sum(dim=1)                   # shape: (1, U, V)
            saliency_map = F.relu(lin_comb)                                 # L^c - shape: (1, U, V)

            saliency_map = saliency_map.unsqueeze(dim=1)                    # shape: (1, 1, U, V)
            align_corners = False if inter_mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None
            saliency_map = F.interpolate(saliency_map,  # shape: (N, 1, H, W)
                                         size=(h, w),
                                         mode=inter_mode,
                                         align_corners=align_corners)
            saliency_map = saliency_map.squeeze(dim=0).squeeze(dim=0)      # shape: (H, W)

            saliency_min = saliency_map.min()
            saliency_max = saliency_map.max()

            saliency_map = (saliency_map - saliency_min) / (saliency_max - saliency_min)
            return saliency_map

    def close(self):
        assert self.entered

        self.forward_handle.remove()
        self.backward_handle.remove()

        self.entered = False

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

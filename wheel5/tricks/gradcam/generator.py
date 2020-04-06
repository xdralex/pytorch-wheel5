import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
import torch.nn.functional as F
from typing import Optional, Callable, Tuple

from wheel5.util import Closure


#
# Inspired by:
#       - https://github.com/1Konny/gradcam_plus_plus-pytorch
#       - https://github.com/frgfm/torch-cam
#


class GradCAMBase(object):
    def __init__(self, model: nn.Module, layer: nn.Module, score_fn: Callable[..., torch.Tensor]):
        self.model = model
        self.layer = layer
        self.score_fn = score_fn

        self.activations_closure = Closure[torch.Tensor]()
        self.gradients_closure = Closure[torch.Tensor]()

        self.forward_handle: Optional[RemovableHandle] = None
        self.backward_handle: Optional[RemovableHandle] = None

        self.entered = False

    @staticmethod
    def upsample(saliency_map: torch.Tensor, inter_mode: str, h: int, w: int) -> torch.Tensor:
        saliency_map = saliency_map.unsqueeze(dim=1)  # shape: (1, 1, U, V)
        align_corners = False if inter_mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None
        saliency_map = F.interpolate(saliency_map,  # shape: (N, 1, H, W)
                                     size=(h, w),
                                     mode=inter_mode,
                                     align_corners=align_corners)
        saliency_map = saliency_map.squeeze(dim=0).squeeze(dim=0)  # shape: (H, W)

        saliency_min = saliency_map.min()
        saliency_max = saliency_map.max()

        saliency_map = (saliency_map - saliency_min) / (saliency_max - saliency_min)
        return saliency_map

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


# https://arxiv.org/pdf/1610.02391.pdf
class GradCAM(GradCAMBase):
    def __init__(self, model: nn.Module, layer: nn.Module, score_fn: Callable[..., torch.Tensor]):
        super(GradCAM, self).__init__(model, layer, score_fn)

    def generate(self, input: torch.Tensor, inter_mode: str = 'bilinear') -> torch.Tensor:
        # input - shape: (1, F, H, W)

        assert len(input.shape) == 4
        assert input.shape[0] == 1

        with torch.enable_grad():
            n, f, h, w = input.shape

            output: torch.Tensor = self.model(input)                        # y - shape: (1, C)
            assert len(output.shape) == 2
            assert output.shape[0] == 1

            score = self.score_fn(output)                                   # y_c - scalar

            assert len(score.shape) == 1
            assert score.shape[0] == n

            self.model.zero_grad()
            score.backward()

            act: torch.Tensor = self.activations_closure.value              # A - shape: (1, K, U, V)
            grad: torch.Tensor = self.gradients_closure.value               # dy^c / dA - shape: (1, K, U, V)

        with torch.no_grad():
            _, k, u, v = grad.shape

            alpha = grad.view(n, k, -1).mean(dim=2)                         # avg over UxV for each A_k - shape: (1, K)
            weights = alpha.view(n, k, 1, 1)                                # α^c_k - shape: (1, K, 1, 1)
            lin_comb = (weights * act).sum(dim=1)                           # shape: (1, U, V)
            saliency_map = F.relu(lin_comb)                                 # L^c - shape: (1, U, V)

            return self.upsample(saliency_map, inter_mode, h, w)


# https://arxiv.org/pdf/1710.11063.pdf
class GradCAMpp(GradCAMBase):
    def __init__(self, model: nn.Module, layer: nn.Module, score_fn: Callable[..., torch.Tensor]):
        super(GradCAMpp, self).__init__(model, layer, score_fn)

    def generate(self, input: torch.Tensor, inter_mode: str = 'bilinear') -> torch.Tensor:
        # input - shape: (1, F, H, W)

        assert len(input.shape) == 4
        assert input.shape[0] == 1

        with torch.enable_grad():
            n, f, h, w = input.shape

            output: torch.Tensor = self.model(input)                        # S - shape: (1, C)
            assert len(output.shape) == 2
            assert output.shape[0] == 1

            score = self.score_fn(output)                                   # S^c - scalar

            assert len(score.shape) == 1
            assert score.shape[0] == n

            self.model.zero_grad()
            score.backward()

            act: torch.Tensor = self.activations_closure.value              # A - shape: (1, K, U, V)
            grad: torch.Tensor = self.gradients_closure.value               # dS^c / dA - shape: (1, K, U, V)

        with torch.no_grad():
            grad_2 = grad.pow(2)
            grad_3 = grad.pow(3)

            alpha_numer = grad_2                                            # (dS^c / dA)**2 - shape: (1, K, U, V)
            alpha_denom = grad_2 * 2 + (grad_3 * act).sum(dim=(2, 3), keepdim=True)

            alpha = alpha_numer / (alpha_denom + 1e-7)                      # α^kc_ij - shape: (1, K, U, V)
            score = score.exp() * grad                                      # dY^c / dA = exp(S^c) * dS^c / dA - shape: (1, K, U, V)
            weights = alpha * F.relu(score).sum(dim=(2, 3), keepdim=True)   # shape: (1, K, 1, 1)
            saliency_map = (weights * act).sum(dim=1)                       # L^c - shape: (1, U, V)

            return self.upsample(saliency_map, inter_mode, h, w)









































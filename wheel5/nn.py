from typing import List

import torch
from torch import nn


class AverageModel(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super(AverageModel, self).__init__()
        self.models = nn.ModuleList(modules=models)

    def forward(self, x):
        y_list = [m(x) for m in self.models]
        return torch.mean(torch.stack(y_list, dim=0), dim=0)
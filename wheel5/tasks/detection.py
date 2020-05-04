from dataclasses import dataclass

import torch


@dataclass
class BoundingBoxes:
    bboxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor

    def __init__(self, bboxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor):
        assert len(bboxes.shape) == 2
        assert bboxes.shape[1] == 4

        assert len(labels.shape) == 1
        assert labels.shape[0] == bboxes.shape[0]

        assert len(scores.shape) == 1
        assert scores.shape[0] == bboxes.shape[0]

        self.n = bboxes.shape[0]
        self.bboxes = bboxes
        self.labels = labels
        self.scores = scores

    def __len__(self):
        return self.n

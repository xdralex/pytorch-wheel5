from dataclasses import dataclass
from struct import pack, unpack, calcsize
from typing import List, Tuple, Dict, Optional

import torch


@dataclass()
class BoundingBox:
    pt_from: Tuple[int, int]
    pt_to: Tuple[int, int]
    label: int
    score: float

    def encode(self) -> bytes:
        return pack('IIIIId', self.pt_from[0], self.pt_from[1], self.pt_to[0], self.pt_to[1], self.label, self.score)

    @staticmethod
    def decode(b: bytes) -> 'BoundingBox':
        x0, y0, x1, y1, label, score = unpack('IIIIId', b)
        return BoundingBox(pt_from=(x0, y0), pt_to=(x1, y1), label=label, score=score)

    @staticmethod
    def size() -> int:
        return calcsize('IIIIId')


def extract_bboxes(result: Dict[str, torch.Tensor],
                   min_score: float = 0.0,
                   top_bboxes: Optional[int] = None,
                   categories: Optional[List[int]] = None) -> List[BoundingBox]:

    result = {key: tensor.cpu() for key, tensor in result.items()}

    bboxes = convert_bboxes(boxes=result['boxes'],
                            labels=result['labels'],
                            scores=result['scores'])

    bboxes = [bbox for bbox in bboxes if bbox.score >= min_score]

    if categories is not None:
        categories_set = set(categories)
        bboxes = [bbox for bbox in bboxes if bbox.label in categories_set]

    if top_bboxes is not None:
        bboxes = sorted(bboxes, key=lambda bbox: -bbox.score)
        bboxes = bboxes[:top_bboxes]

    return bboxes


def convert_bboxes(boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor) -> List[BoundingBox]:
    assert len(boxes.shape) == 2
    assert boxes.shape[1] == 4

    assert len(labels.shape) == 1
    assert labels.shape[0] == boxes.shape[0]

    assert len(scores.shape) == 1
    assert scores.shape[0] == boxes.shape[0]

    n = boxes.shape[0]

    boxes = boxes.int().numpy()
    labels = labels.numpy()
    scores = scores.numpy()

    entries = []
    for i in range(0, n):
        entries.append(BoundingBox(
            pt_from=(int(boxes[i][0]), int(boxes[i][1])),
            pt_to=(int(boxes[i][2]), int(boxes[i][3])),
            label=int(labels[i]),
            score=float(scores[i])
        ))

    return entries

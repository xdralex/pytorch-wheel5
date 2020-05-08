import math
from dataclasses import dataclass
from struct import pack, unpack, calcsize
from typing import List, Tuple, Optional, Union

import torch


@dataclass
class Rectangle:
    pt_from: Tuple[int, int]
    pt_to: Tuple[int, int]

    def __init__(self, pt_from: Tuple[int, int], pt_to: Tuple[int, int]):
        assert pt_from[0] <= pt_to[0]
        assert pt_from[1] <= pt_to[1]
        self.pt_from = pt_from
        self.pt_to = pt_to

    @property
    def x0(self) -> int:
        return self.pt_from[0]

    @property
    def y0(self) -> int:
        return self.pt_from[1]

    @property
    def x1(self) -> int:
        return self.pt_to[0]

    @property
    def y1(self) -> int:
        return self.pt_to[1]

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    def intersection(self, other: 'Rectangle') -> Optional['Rectangle']:
        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)

        if x0 >= x1 or y0 >= y1:
            return None
        else:
            return Rectangle(pt_from=(x0, y0), pt_to=(x1, y1))

    def iou(self, other: 'Rectangle') -> float:
        intersection = self.intersection(other)
        if intersection is None:
            return 0
        else:
            return intersection.area / (self.area + other.area - intersection.area)

    def overlap(self, other: 'Rectangle') -> float:
        intersection = self.intersection(other)
        if intersection is None:
            return 0
        else:
            return intersection.area / min(self.area, other.area)

    def expand(self, coeff: Union[float, Tuple[float, float]]) -> 'Rectangle':
        if isinstance(coeff, float):
            w_c, h_c = coeff, coeff
        elif isinstance(coeff, tuple):
            w_c, h_c = coeff
        else:
            raise ValueError(f'Expansion coefficient must be either float or (float, float)')

        new_width = self.width * (1 + w_c)
        new_height = self.height * (1 + h_c)

        center_x = (self.x0 + self.x1) / 2
        center_y = (self.y0 + self.y1) / 2

        new_x0 = round(center_x - new_width / 2)
        new_x1 = round(center_x + new_width / 2)
        new_y0 = round(center_y - new_height / 2)
        new_y1 = round(center_y + new_height / 2)

        return Rectangle(pt_from=(new_x0, new_y0), pt_to=(new_x1, new_y1))


@dataclass
class BoundingBox(Rectangle):
    label: int
    score: float

    def __init__(self, pt_from: Tuple[int, int], pt_to: Tuple[int, int], label: int, score: float):
        super(BoundingBox, self).__init__(pt_from, pt_to)
        self.label = label
        self.score = score

    def encode(self) -> bytes:
        return pack('IIIIId', self.pt_from[0], self.pt_from[1], self.pt_to[0], self.pt_to[1], self.label, self.score)

    @staticmethod
    def decode(b: bytes) -> 'BoundingBox':
        x0, y0, x1, y1, label, score = unpack('IIIIId', b)
        return BoundingBox(pt_from=(x0, y0), pt_to=(x1, y1), label=label, score=score)

    @staticmethod
    def byte_size() -> int:
        return calcsize('IIIIId')

    def intersection(self, other: 'Rectangle') -> Optional['BoundingBox']:
        rect = super(BoundingBox, self).intersection(other)
        if rect is None:
            return None
        else:
            return BoundingBox(rect.pt_from, rect.pt_to, self.label, self.score)

    def expand(self, coeff: Union[float, Tuple[float, float]]) -> 'BoundingBox':
        rect = super(BoundingBox, self).expand(coeff)
        return BoundingBox(rect.pt_from, rect.pt_to, self.label, self.score)


def non_maximum_suppression(bboxes: List[BoundingBox],
                            threshold: float,
                            ranking: str = 'score_sqrt_area',
                            suppression: str = 'overlap') -> List[BoundingBox]:

    if ranking == 'score':
        bboxes = sorted(bboxes, key=lambda bbox: -bbox.score)
    elif ranking == 'score_area':
        bboxes = sorted(bboxes, key=lambda bbox: -bbox.score * bbox.area)
    elif ranking == 'score_sqrt_area':
        bboxes = sorted(bboxes, key=lambda bbox: -bbox.score * math.sqrt(bbox.area))
    elif ranking == 'score_log_area':
        bboxes = sorted(bboxes, key=lambda bbox: -bbox.score * (0 if bbox.area <= 1 else math.log2(bbox.area)))
    else:
        raise ValueError(f'Ranking mode {ranking} is not supported')

    result = []
    bboxes_temp = []
    while len(bboxes) > 0:
        best = bboxes[0]
        result.append(best)

        for candidate in bboxes[1:]:
            if suppression == 'iou':
                state = best.iou(candidate)
            elif suppression == 'overlap':
                state = best.overlap(candidate)
            else:
                raise ValueError(f'Suppression mode {suppression} is not supported')

            if state <= threshold:
                bboxes_temp.append(candidate)

        bboxes, bboxes_temp = bboxes_temp, []

    return result


# TODO: top_bboxes should be removed as it conflicts with NMS logic
def filter_bboxes(bboxes: List[BoundingBox],
                  min_score: float = 0.0,
                  top_bboxes: Optional[int] = None,
                  categories: Optional[List[int]] = None) -> List[BoundingBox]:
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

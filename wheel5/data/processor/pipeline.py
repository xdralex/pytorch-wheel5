from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .functional import mixup, cutmix


class TargetFormat(Enum):
    CLASS_INDEX = 1
    SOFT_LABEL = 2


class DataIterator(ABC):
    def __iter__(self) -> 'DataIterator':
        return self

    @abstractmethod
    def __next__(self) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        pass


class DataProcessor(ABC):
    @abstractmethod
    def __iter__(self) -> DataIterator:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass

    @property
    @abstractmethod
    def target_format(self) -> TargetFormat:
        pass


class DirectDataIterator(DataIterator):
    def __init__(self, iterator):
        self.iterator = iterator

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor]:
        return next(self.iterator)


class DirectDataProcessor(DataProcessor):
    def __init__(self, loader: DataLoader, target_format: TargetFormat):
        self.loader = loader
        self._target_format = target_format

    def __iter__(self) -> DataIterator:
        return DirectDataIterator(iter(self.loader))

    def __len__(self) -> int:
        return len(self.loader.sampler)

    @property
    def batch_size(self) -> int:
        return self.loader.batch_size

    @property
    def dataset(self) -> Dataset:
        return self.loader.dataset

    @property
    def target_format(self) -> TargetFormat:
        return self._target_format


class OneHotDataIterator(DataIterator):
    def __init__(self, iterator, num_classes: int):
        self.iterator = iterator
        self.num_classes = num_classes

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor]:
        img, lb, indices = next(self.iterator)

        with torch.no_grad():
            lb = F.one_hot(lb, self.num_classes).type_as(img)

            # transforming lb shape from (N, d_1, d_2, ..., d_K, C) to (N, C, d_1, d_2, ..., d_K)
            order = list(range(0, lb.ndim))
            order[1], order[-1] = order[-1], order[1]
            lb = lb.permute(order)

            return img, lb, indices


class OneHotDataProcessor(DataProcessor):
    def __init__(self, processor: DataProcessor, num_classes: int):
        assert processor.target_format == TargetFormat.CLASS_INDEX

        self.processor = processor
        self.num_classes = num_classes

    def __iter__(self) -> DataIterator:
        return OneHotDataIterator(iter(self.processor), num_classes=self.num_classes)

    def __len__(self) -> int:
        return len(self.processor)

    @property
    def batch_size(self) -> int:
        return self.processor.batch_size

    @property
    def target_format(self) -> TargetFormat:
        return TargetFormat.SOFT_LABEL


class TwoMixDataIterator(DataIterator):
    def __init__(self, iterator1, iterator2, mixer_fn):
        self.iterator1 = iterator1
        self.iterator2 = iterator2
        self.mixer_fn = mixer_fn

    def __next__(self) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        img1, lb1, indices1 = next(self.iterator1)
        img2, lb2, indices2 = next(self.iterator2)

        assert indices1.shape == indices2.shape
        img, lb = self.mixer_fn(img1, lb1, img2, lb2)

        return img, lb, None


class TwoMixDataProcessor(DataProcessor):
    def __init__(self, processor1: DataProcessor, processor2: DataProcessor, mixer_fn):
        assert processor1.target_format == TargetFormat.SOFT_LABEL
        assert processor2.target_format == TargetFormat.SOFT_LABEL

        self.processor1 = processor1
        self.processor2 = processor2
        self.mixer_fn = mixer_fn

    def __iter__(self) -> DataIterator:
        return TwoMixDataIterator(iter(self.processor1), iter(self.processor2), self.mixer_fn)

    def __len__(self) -> int:
        len_1 = len(self.processor1)
        len_2 = len(self.processor2)

        assert len_1 == len_2
        return len_1

    @property
    def batch_size(self) -> int:
        batch_size1 = self.processor1.batch_size
        batch_size2 = self.processor2.batch_size

        assert batch_size1 == batch_size2
        return batch_size1

    @property
    def target_format(self) -> TargetFormat:
        return TargetFormat.SOFT_LABEL


class MixupDataProcessor(TwoMixDataProcessor):
    def __init__(self, processor1: DataProcessor, processor2: DataProcessor, alpha: float):
        def mixer_fn(img1: Tensor, lb1: Tensor, img2: Tensor, lb2: Tensor):
            return mixup(img1, lb1, img2, lb2, alpha)

        super(MixupDataProcessor, self).__init__(processor1, processor2, mixer_fn)


class CutMixDataProcessor(TwoMixDataProcessor):
    def __init__(self, processor1: DataProcessor, processor2: DataProcessor, alpha: float, mode: str = 'compact'):
        def mixer_fn(img1: Tensor, lb1: Tensor, img2: Tensor, lb2: Tensor):
            return cutmix(img1, lb1, img2, lb2, alpha, mode)

        super(CutMixDataProcessor, self).__init__(processor1, processor2, mixer_fn)

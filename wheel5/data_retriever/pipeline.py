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


class DataRetriever(ABC):
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


class DirectDataRetriever(DataRetriever):
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


class OneHotDataRetriever(DataRetriever):
    def __init__(self, retriever: DataRetriever, num_classes: int):
        assert retriever.target_format == TargetFormat.CLASS_INDEX

        self.retriever = retriever
        self.num_classes = num_classes

    def __iter__(self) -> DataIterator:
        return OneHotDataIterator(iter(self.retriever), num_classes=self.num_classes)

    def __len__(self) -> int:
        return len(self.retriever)

    @property
    def batch_size(self) -> int:
        return self.retriever.batch_size

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


class TwoMixDataRetriever(DataRetriever):
    def __init__(self, retriever1: DataRetriever, retriever2: DataRetriever, mixer_fn):
        assert retriever1.target_format == TargetFormat.SOFT_LABEL
        assert retriever2.target_format == TargetFormat.SOFT_LABEL

        self.retriever1 = retriever1
        self.retriever2 = retriever2
        self.mixer_fn = mixer_fn

    def __iter__(self) -> DataIterator:
        return TwoMixDataIterator(iter(self.retriever1), iter(self.retriever2), self.mixer_fn)

    def __len__(self) -> int:
        len_1 = len(self.retriever1)
        len_2 = len(self.retriever2)

        assert len_1 == len_2
        return len_1

    @property
    def batch_size(self) -> int:
        batch_size1 = self.retriever1.batch_size
        batch_size2 = self.retriever2.batch_size

        assert batch_size1 == batch_size2
        return batch_size1

    @property
    def target_format(self) -> TargetFormat:
        return TargetFormat.SOFT_LABEL


class MixupDataRetriever(TwoMixDataRetriever):
    def __init__(self, retriever1: DataRetriever, retriever2: DataRetriever, alpha: float):
        def mixer_fn(img1: Tensor, lb1: Tensor, img2: Tensor, lb2: Tensor):
            return mixup(img1, lb1, img2, lb2, alpha)

        super(MixupDataRetriever, self).__init__(retriever1, retriever2, mixer_fn)


class CutMixDataRetriever(TwoMixDataRetriever):
    def __init__(self, retriever1: DataRetriever, retriever2: DataRetriever, alpha: float, mode: str = 'compact'):
        def mixer_fn(img1: Tensor, lb1: Tensor, img2: Tensor, lb2: Tensor):
            return cutmix(img1, lb1, img2, lb2, alpha, mode)

        super(CutMixDataRetriever, self).__init__(retriever1, retriever2, mixer_fn)

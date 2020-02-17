from abc import ABC, abstractmethod
from typing import Tuple, Callable, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F


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


class DirectDataIterator(DataIterator):
    def __init__(self, loader_iter):
        self.loader_iter = loader_iter

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor]:
        return next(self.loader_iter)


class DirectDataRetriever(DataRetriever):
    def __init__(self, loader: DataLoader):
        self.loader = loader

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


class MixupDataIterator(DataIterator):
    def __init__(self, loader_iter1, loader_iter2, num_classes: int, mix_sampler: Callable[[], float]):
        self.loader_iter1 = loader_iter1
        self.loader_iter2 = loader_iter2
        self.num_classes = num_classes
        self.mix_sampler = mix_sampler

    def __next__(self) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        x1, y1, indices1 = next(self.loader_iter1)
        x2, y2, indices2 = next(self.loader_iter2)

        with torch.no_grad():
            assert x1.shape == x2.shape
            assert y1.shape == y2.shape
            assert indices1.shape == indices2.shape

            # transforming targets to one hot shape
            y1 = F.one_hot(y1, self.num_classes).type_as(x1)
            y2 = F.one_hot(y2, self.num_classes).type_as(x2)

            # mixing inputs and targets
            q = self.mix_sampler()
            x = torch.lerp(x1, x2, weight=q)
            y = torch.lerp(y1, y2, weight=q)

            # transforming y shape: (N, d_1, d_2, ..., d_K, C) => (N, C, d_1, d_2, ..., d_K)
            order = list(range(0, y.ndim))
            order[1], order[-1] = order[-1], order[1]
            y = y.permute(order)

            return x, y, None


class MixupDataRetriever(DataRetriever):
    def __init__(self, loader1: DataLoader, loader2: DataLoader, num_classes: int, mix_sampler: Callable[[], float]):
        self.loader1 = loader1
        self.loader2 = loader2
        self.num_classes = num_classes
        self.mix_sampler = mix_sampler

    def __iter__(self) -> DataIterator:
        return MixupDataIterator(iter(self.loader1), iter(self.loader2), self.num_classes, self.mix_sampler)

    def __len__(self) -> int:
        len_1 = len(self.loader1.sampler)
        len_2 = len(self.loader2.sampler)

        assert len_1 == len_2
        return len_1

    @property
    def batch_size(self) -> int:
        batch_size1 = self.loader1.batch_size
        batch_size2 = self.loader2.batch_size

        assert batch_size1 == batch_size2
        return batch_size1

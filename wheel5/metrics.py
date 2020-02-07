from abc import ABC, abstractmethod

import torch
from numpy.random.mtrand import RandomState
from torch import Tensor


class Meter(ABC):
    @abstractmethod
    def add(self, *args):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def value(self):
        pass


class LastMeter(Meter):
    def __init__(self):
        self.last = None

    def add(self, value):
        self.last = value
        return value

    def reset(self):
        self.last = None

    def value(self):
        return self.last


class CountMeter(Meter):
    def __init__(self):
        self.count = 0

    def add(self):
        self.count += 1
        return 1

    def reset(self):
        self.count = 0

    def value(self):
        return self.count


class SumMeter(Meter):
    def __init__(self):
        self.sum = 0.0

    def add(self, value):
        self.sum += value
        return value

    def reset(self):
        self.sum = 0.0

    def value(self):
        return self.sum


class AverageMeter(Meter):
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def add(self, value):
        self.sum += value
        self.count += 1
        return value

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def value(self):
        if self.count == 0:
            return None
        else:
            return self.sum / self.count


class MaxMeter(Meter):
    def __init__(self):
        self.best = None

    def add(self, value):
        if self.best is None:
            self.best = value
        else:
            self.best = max(self.best, value)
        return value

    def reset(self):
        self.best = None

    def value(self):
        return self.best


class MinMeter(Meter):
    def __init__(self):
        self.best = None

    def add(self, value):
        if self.best is None:
            self.best = value
        else:
            self.best = min(self.best, value)
        return value

    def reset(self):
        self.best = None

    def value(self):
        return self.best


class AccuracyMeter(Meter):
    def __init__(self):
        self.correct_sum = 0.0
        self.total_sum = 0.0

    def add(self, correct, total):
        self.correct_sum += correct
        self.total_sum += total
        return correct / total

    def reset(self):
        self.correct_sum = 0.0
        self.total_sum = 0.0

    def value(self):
        return self.correct_sum / self.total_sum


class ArrayAccumMeter(Meter):
    def __init__(self):
        self.accum = []

    def add(self, tensor: Tensor):
        self.accum.append(tensor)

    def reset(self):
        self.accum = []

    def value(self):
        return torch.cat(self.accum)


class ReservoirSamplingMeter(Meter):
    def __init__(self, k, random_state: RandomState = None):
        self.k = k
        self.random_state = random_state or RandomState()

        self.accum = []
        self.counter = 0

    def add(self, elements):
        for element in elements:
            if len(self.accum) < self.k:
                self.accum.append(element)
            else:
                r = self.random_state.randint(0, self.counter + 1)
                if r < self.k:
                    self.accum[r] = element

            self.counter += 1

    def reset(self):
        self.accum = []
        self.counter = 0

    def value(self):
        return self.accum

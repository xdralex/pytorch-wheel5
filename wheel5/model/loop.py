import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, NamedTuple, Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import log_softmax
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..cuda import memory_stats
from ..metering import AverageMeter, AccuracyMeter, ArrayAccumMeter, ReservoirSamplingMeter, LimitedSamplingMeter


class TrainEvalSample(NamedTuple):
    x: Tensor
    y: Tensor
    z: Tensor
    y_probs: Tensor
    y_hat: Tensor


class Predictions(NamedTuple):
    y: Tensor
    z: Tensor
    y_probs: Tensor
    y_hat: Tensor
    indices: Tensor


class EpochHandler(ABC):
    @abstractmethod
    def epoch_started(self):
        pass

    @abstractmethod
    def batch_processed(self, x: Tensor, y: Tensor, z: Tensor, loss_value: Optional[Tensor], indices: Optional[Tensor]):
        pass

    @abstractmethod
    def epoch_finished(self):
        pass

    @abstractmethod
    def state_repr(self) -> str:
        pass


class TrainEvalEpochHandler(EpochHandler):
    def __init__(self, kind, num_epochs, accuracy: 'Accuracy', sampled_epochs=-1, samples=8):
        self.accuracy = accuracy

        self.loss_meter = AverageMeter()
        self.accuracy_meter = AccuracyMeter()

        self.sampled_epochs = sampled_epochs
        self.random_samples_meter = ReservoirSamplingMeter(k=samples)
        self.fixed_samples_meter = LimitedSamplingMeter(k=samples)

        self.kind = kind
        self.epoch = 0
        self.in_epoch = False
        self.num_epochs = num_epochs

        self._state_repr = ''

    def epoch_started(self):
        assert 0 <= self.epoch < self.num_epochs
        assert not self.in_epoch
        self.epoch += 1
        self.in_epoch = True

        self.loss_meter.reset()
        self.accuracy_meter.reset()

        self.random_samples_meter.reset()
        self.fixed_samples_meter.reset()

        self._state_repr = f'{self._prefix()} - loss=?, acc=?'

    def batch_processed(self, x: Tensor, y: Tensor, z: Tensor, loss_value: Tensor, indices: Optional[Tensor]):
        assert 0 < self.epoch <= self.num_epochs
        assert self.in_epoch

        x_cpu = x.cpu()
        y_cpu = y.cpu()
        z_cpu = z.cpu()
        y_probs = torch.exp(log_softmax(z_cpu, dim=1))
        y_hat = torch.argmax(y_probs, dim=1)

        correct, total = self.accuracy(y_cpu, z_cpu, y_probs, y_hat)

        batch_loss = self.loss_meter.add(float(loss_value))
        batch_accuracy = self.accuracy_meter.add(correct, total)

        if self.epoch <= self.sampled_epochs:
            elements = []
            for i in range(0, y_cpu.shape[0]):
                sample = TrainEvalSample(x=x_cpu[i], y=y_cpu[i], z=z_cpu[i], y_probs=y_probs[i], y_hat=y_hat[i])
                elements.append(sample)

            self.random_samples_meter.add(elements)
            self.fixed_samples_meter.add(elements)

        self._state_repr = f'{self._prefix()} - accuracy={batch_accuracy:.6f}; loss={batch_loss:.6f}'

    def epoch_finished(self) -> Dict[str, Union[int, float]]:
        assert 0 < self.epoch <= self.num_epochs
        assert self.in_epoch
        self.in_epoch = False

        epoch_loss = self.loss_meter.value()
        epoch_accuracy = self.accuracy_meter.value()

        self._state_repr = f'{self._prefix()} - accuracy={epoch_accuracy:.6f}; loss={epoch_loss:.6f}'
        return {'acc': epoch_accuracy, 'loss': epoch_loss}

    def state_repr(self) -> str:
        return self._state_repr

    def _prefix(self) -> str:
        ep_width = int(np.ceil(np.log10(self.num_epochs + 1)))
        return f'[{self.epoch:>{ep_width}} / {self.num_epochs}] {self.kind}'


class PredictEpochHandler(EpochHandler):
    def __init__(self):
        self.y_accum = ArrayAccumMeter()
        self.z_accum = ArrayAccumMeter()
        self.indices_accum = ArrayAccumMeter()

        self.in_epoch = False

    def epoch_started(self):
        assert not self.in_epoch
        self.in_epoch = True

        self.y_accum.reset()
        self.z_accum.reset()
        self.indices_accum.reset()

    def batch_processed(self, x: Tensor, y: Tensor, z: Tensor, loss_value: Optional[Tensor], indices: Tensor):
        assert self.in_epoch

        self.y_accum.add(y.cpu())
        self.z_accum.add(z.cpu())
        self.indices_accum.add(indices.cpu())

    def epoch_finished(self) -> Predictions:
        assert self.in_epoch
        self.in_epoch = False

        y = self.y_accum.value()
        z = self.z_accum.value()
        y_probs = torch.nn.functional.log_softmax(z, dim=1)
        y_hat = torch.argmax(y_probs, dim=1)
        indices = self.indices_accum.value()

        return Predictions(y=y, z=z, y_probs=y_probs, y_hat=y_hat, indices=indices)

    def state_repr(self) -> str:
        return 'predict'


def run_epoch(device: Union[torch.device, int],
              model: Module,
              loader: DataLoader,
              loss: Optional[Module],
              optimizer: Optional[Optimizer],
              scheduler: Optional[Any],
              handler: EpochHandler,
              display_progress: bool = True):
    logger = logging.getLogger(f'{__name__}.run_epoch')

    def log_status(context: str):
        if logger.getEffectiveLevel() == logging.DEBUG:
            stats = memory_stats(device)
            logger.debug(f'{context} - [{device}] alloc/cache = {stats["allocated"]:.0f} MB / {stats["cached"]:.0f} MB')

    batches_count = len(loader)
    train_mode = optimizer is not None

    model.train(train_mode)
    with torch.autograd.set_grad_enabled(train_mode):
        with tqdm(total=batches_count, disable=not display_progress) as progress_bar:
            log_status('started epoch')

            handler.epoch_started()
            progress_bar.set_description(handler.state_repr())

            for i, (x_cpu, y_cpu, indices) in enumerate(loader):
                log_status('  started batch')

                x = x_cpu.to(device)
                y = y_cpu.to(device)
                log_status('    loaded batch')

                z = model(x)
                log_status('    performed forward pass')

                loss_value = None if loss is None else loss(z, y)
                log_status('    computed loss value')

                if optimizer is not None:
                    if loss_value is None:
                        raise AssertionError('Loss function must be provided if the optimizer is set')

                    optimizer.zero_grad()
                    loss_value.backward()
                    log_status('    performed backward pass')

                    optimizer.step()
                    log_status('    stepped optimizer')

                handler.batch_processed(x, y, z, loss_value, indices)
                progress_bar.update()
                progress_bar.set_description(handler.state_repr())

                log_status('    finished batch')

            if scheduler is not None:
                scheduler.step()
                log_status('stepped scheduler')

            results = handler.epoch_finished()
            progress_bar.set_description(handler.state_repr())

            log_status('finished epoch')

            return results

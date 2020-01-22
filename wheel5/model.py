import collections
import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, List

import numpy as np
import torch
from pandas import DataFrame
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import cuda
from .metrics import AverageMeter, AccuracyMeter, ArrayAccumMeter
from .snapshotters import Snapshotter, FitState


class EpochHandler(ABC):
    @abstractmethod
    def epoch_started(self):
        pass

    @abstractmethod
    def batch_processed(self, x: Tensor, y: Tensor, y_probs: Tensor, y_hat: Tensor, loss_value: Optional[Tensor], indices: Tensor):
        pass

    @abstractmethod
    def epoch_finished(self):
        pass

    @abstractmethod
    def state_repr(self) -> str:
        pass


class TrainEvalEpochHandler(EpochHandler):
    def __init__(self, kind, num_epochs):
        self.loss_meter = AverageMeter()
        self.accuracy_meter = AccuracyMeter()

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

        self._state_repr = f'{self._prefix()} - loss=?, accuracy=?'

    def batch_processed(self, x: Tensor, y: Tensor, y_probs: Tensor, y_hat: Tensor, loss_value: Tensor, indices: Tensor):
        assert 0 < self.epoch <= self.num_epochs
        assert self.in_epoch

        batch_correct = float(torch.sum(y_hat == y))
        batch_total = float(y.shape[0])

        batch_loss = self.loss_meter.add(float(loss_value))
        batch_accuracy = self.accuracy_meter.add(batch_correct, batch_total)

        self._state_repr = f'{self._prefix()} - loss={batch_loss:.6f}, accuracy={batch_accuracy:.3f}'

    def epoch_finished(self) -> Dict[str, Union[int, float]]:
        assert 0 < self.epoch <= self.num_epochs
        assert self.in_epoch
        self.in_epoch = False

        epoch_loss = self.loss_meter.value()
        epoch_accuracy = self.accuracy_meter.value()

        self._state_repr = f'{self._prefix()} - loss={epoch_loss:.6f}, accuracy={epoch_accuracy:.3f}'
        return {'loss': epoch_loss, 'accuracy': epoch_accuracy}

    def state_repr(self) -> str:
        return self._state_repr

    def _prefix(self) -> str:
        ep_width = int(np.ceil(np.log10(self.num_epochs + 1)))
        return f'[{self.epoch:>{ep_width}} / {self.num_epochs}] {self.kind}'


class PredictEpochHandler(EpochHandler):
    def __init__(self):
        self.y_accum = ArrayAccumMeter()
        self.y_probs_accum = ArrayAccumMeter()
        self.y_hat_accum = ArrayAccumMeter()
        self.indices_accum = ArrayAccumMeter()

        self.in_epoch = False

    def epoch_started(self):
        assert not self.in_epoch
        self.in_epoch = True

        self.y_accum.reset()
        self.y_probs_accum.reset()
        self.y_hat_accum.reset()
        self.indices_accum.reset()

    def batch_processed(self, x: Tensor, y: Tensor, y_probs: Tensor, y_hat: Tensor, loss_value: Optional[Tensor], indices: Tensor):
        assert self.in_epoch

        self.y_accum.add(y.cpu().numpy())
        self.y_probs_accum.add(y_probs.cpu().numpy())
        self.y_hat_accum.add(y_hat.cpu().numpy())
        self.indices_accum.add(indices.cpu().numpy())

    def epoch_finished(self) -> Dict[str, np.ndarray]:
        assert self.in_epoch
        self.in_epoch = False

        return {
            'y': self.y_accum.value(),
            'y_probs': self.y_probs_accum.value(),
            'y_hat': self.y_hat_accum.value(),
            'indices': self.indices_accum.value()
        }

    def state_repr(self) -> str:
        return 'predict'


def fit(device: Union[torch.device, int],
        model: Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss: Module,
        optimizer: Optimizer,
        num_epochs: int,
        snapshotter: Optional[Union[Snapshotter, List[Snapshotter]]] = None,
        tb_writer: Optional[SummaryWriter] = None,
        display_progress: bool = True) -> DataFrame:

    if snapshotter is None:
        snapshotter = []
    elif not isinstance(snapshotter, collections.Iterable):
        snapshotter = [snapshotter]

    train_handler = TrainEvalEpochHandler('train', num_epochs)
    val_handler = TrainEvalEpochHandler('val', num_epochs)

    metrics_list = []

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch(device, model, train_loader, loss, optimizer, train_handler, display_progress=display_progress)
        val_metrics = run_epoch(device, model, val_loader, loss, None, val_handler, display_progress=display_progress)

        assert set(train_metrics.keys()) == set(val_metrics.keys())
        assert 'epoch' not in train_metrics

        metrics = {'epoch': epoch}
        for key in train_metrics.keys():
            tb_writer.add_scalar(f'fit/train/{key}', train_metrics[key], epoch)
            tb_writer.add_scalar(f'fit/val/{key}', val_metrics[key], epoch)

            metrics[f'train_{key}'] = train_metrics[key]
            metrics[f'val_{key}'] = val_metrics[key]

        tb_writer.flush()
        metrics_list.append(metrics)

        state = FitState(
            model=model,
            loss=loss,
            optimizer=optimizer,
            epoch=epoch,
            num_epochs=num_epochs,
            train_metrics=train_metrics,
            val_metrics=val_metrics
        )

        for entry in snapshotter:
            entry.epoch_completed(state)

    return DataFrame(metrics_list)


def score(device: Union[torch.device, int],
          model: Module,
          loader: DataLoader,
          loss: Module,
          display_progress: bool = True) -> Dict[str, Union[int, float]]:

    handler = TrainEvalEpochHandler('score', num_epochs=1)
    return run_epoch(device, model, loader, loss, None, handler, display_progress=display_progress)


def predict(device: Union[torch.device, int],
            model: Module,
            loader: DataLoader,
            display_progress: bool = True) -> Dict[str, np.ndarray]:

    handler = PredictEpochHandler()
    return run_epoch(device, model, loader, None, None, handler, display_progress=display_progress)


def run_epoch(device: Union[torch.device, int],
              model: Module,
              loader: DataLoader,
              loss: Optional[Module],
              optimizer: Optional[Optimizer],
              handler: EpochHandler,
              display_progress: bool = True):

    logger = logging.getLogger(f'{__name__}.run_epoch')

    def log_memory_usage(context: str):
        if logger.getEffectiveLevel() == logging.DEBUG:
            stats = cuda.memory_stats(device)
            logger.debug(f'{context} - [{device}] alloc/cache = {stats["allocated"]:.0f} MB / {stats["cached"]:.0f} MB')

    batches_count = math.ceil(len(loader.sampler) / loader.batch_size)
    train_mode = optimizer is not None

    model.train(train_mode)
    with torch.autograd.set_grad_enabled(train_mode):
        with tqdm(total=batches_count, disable=not display_progress) as progress_bar:
            log_memory_usage('started epoch')

            handler.epoch_started()
            progress_bar.set_description(handler.state_repr())

            for i, (x, y, _, indices) in enumerate(loader):
                log_memory_usage('  started batch')

                x_gpu = x.to(device)
                y_gpu = y.to(device)
                log_memory_usage('    loaded batch')

                y_probs = model(x_gpu)
                y_hat = torch.argmax(y_probs, 1)
                log_memory_usage('    performed forward pass')

                loss_value = None if loss is None else loss(y_probs, y_gpu)
                log_memory_usage('    computed loss value')

                if optimizer is not None:
                    if loss_value is None:
                        raise AssertionError('Loss function must be provided if the optimizer is set')

                    optimizer.zero_grad()
                    loss_value.backward()
                    log_memory_usage('    performed backward pass')

                    optimizer.step()
                    log_memory_usage('    stepped optimizer')

                handler.batch_processed(x_gpu, y_gpu, y_probs, y_hat, loss_value, indices)
                progress_bar.update()
                progress_bar.set_description(handler.state_repr())

                log_memory_usage('    finished batch')

            metrics = handler.epoch_finished()
            progress_bar.set_description(handler.state_repr())

            log_memory_usage('finished epoch')

            return metrics

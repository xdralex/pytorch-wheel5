import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import cuda
from .metrics import AverageMeter, AccuracyMeter, ArrayAccumMeter
from .tracking import TrialTracker, FitState


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
        self.acc_meter = AccuracyMeter()

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
        self.acc_meter.reset()

        self._state_repr = f'{self._prefix()} - loss=?, acc=?'

    def batch_processed(self, x: Tensor, y: Tensor, y_probs: Tensor, y_hat: Tensor, loss_value: Tensor, indices: Tensor):
        assert 0 < self.epoch <= self.num_epochs
        assert self.in_epoch

        batch_correct = float(torch.sum(y_hat == y))
        batch_total = float(y.shape[0])

        batch_loss = self.loss_meter.add(float(loss_value))
        batch_acc = self.acc_meter.add(batch_correct, batch_total)

        self._state_repr = f'{self._prefix()} - loss={batch_loss:.6f}, acc={batch_acc:.3f}'

    def epoch_finished(self) -> Dict[str, Union[int, float]]:
        assert 0 < self.epoch <= self.num_epochs
        assert self.in_epoch
        self.in_epoch = False

        epoch_loss = self.loss_meter.value()
        epoch_acc = self.acc_meter.value()

        self._state_repr = f'{self._prefix()} - loss={epoch_loss:.6f}, acc={epoch_acc:.3f}'
        return {'loss': epoch_loss, 'acc': epoch_acc}

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

        self.y_accum.add(y.cpu())
        self.y_probs_accum.add(y_probs.cpu())
        self.y_hat_accum.add(y_hat.cpu())
        self.indices_accum.add(indices.cpu())

    def epoch_finished(self) -> Dict[str, Tensor]:
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
        tracker: Optional[TrialTracker] = None,
        display_progress: bool = True):
    train_handler = TrainEvalEpochHandler('train', num_epochs)
    val_handler = TrainEvalEpochHandler('val', num_epochs)

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch(device, model, train_loader, loss, optimizer, train_handler, display_progress=display_progress)
        val_metrics = run_epoch(device, model, val_loader, loss, None, val_handler, display_progress=display_progress)

        if tracker:
            state = FitState(model=model,
                             loss=loss,
                             optimizer=optimizer,
                             epoch=epoch,
                             num_epochs=num_epochs,
                             train_metrics=train_metrics,
                             val_metrics=val_metrics)

            tracker.epoch_completed(state)


def score(device: Union[torch.device, int],
          model: Module,
          loader: DataLoader,
          loss: Module,
          display_progress: bool = True) -> Dict[str, Union[int, float]]:
    handler = TrainEvalEpochHandler('score', num_epochs=1)
    return run_epoch(device, model, loader, loss, None, handler, display_progress=display_progress)


def score_blend(device: Union[torch.device, int],
                models: List[Module],
                loader: DataLoader,
                loss: Module,
                display_progress: bool = True) -> Dict[str, Union[int, float]]:

    assert len(models) > 0

    y = None
    y_probs_list = []

    for model in models:
        model_device = model.to(device)

        handler = PredictEpochHandler()
        results = run_epoch(device, model_device, loader, None, None, handler, display_progress=display_progress)

        order = torch.argsort(results['indices'])
        y_ordered = torch.index_select(results['y'], dim=0, index=order)
        y_probs_ordered = torch.index_select(results['y_probs'], dim=0, index=order)

        if y is None:
            y = y_ordered
        else:
            assert bool(torch.eq(y, y_ordered).all())

        y_probs_list.append(y_probs_ordered)

        del model

    y_probs_stack = torch.stack(y_probs_list, dim=0)
    y_probs_blend = torch.mean(y_probs_stack, dim=0)

    y_hat = torch.argmax(y_probs_blend, 1)
    loss_value = float(loss(y_probs_blend, y))

    correct = float(torch.sum(y_hat == y))
    total = float(y.shape[0])

    acc = correct / total

    return {'loss': loss_value, 'acc': acc}


def predict(device: Union[torch.device, int],
            model: Module,
            loader: DataLoader,
            display_progress: bool = True) -> Dict[str, Tensor]:
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

    def log_status(context: str):
        if logger.getEffectiveLevel() == logging.DEBUG:
            stats = cuda.memory_stats(device)
            logger.debug(f'{context} - [{device}] alloc/cache = {stats["allocated"]:.0f} MB / {stats["cached"]:.0f} MB')

    batches_count = math.ceil(len(loader.sampler) / loader.batch_size)
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

                y_probs = model(x)
                y_hat = torch.argmax(y_probs, 1)
                log_status('    performed forward pass')

                loss_value = None if loss is None else loss(y_probs, y)
                log_status('    computed loss value')

                if optimizer is not None:
                    if loss_value is None:
                        raise AssertionError('Loss function must be provided if the optimizer is set')

                    optimizer.zero_grad()
                    loss_value.backward()
                    log_status('    performed backward pass')

                    optimizer.step()
                    log_status('    stepped optimizer')

                handler.batch_processed(x, y, y_probs, y_hat, loss_value, indices)
                progress_bar.update()
                progress_bar.set_description(handler.state_repr())

                log_status('    finished batch')

            metrics = handler.epoch_finished()
            progress_bar.set_description(handler.state_repr())

            log_status('finished epoch')

            return metrics

import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, List, NamedTuple, Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from . import cuda
from .dataretriever import DataRetriever, DirectDataRetriever
from .metrics import AverageMeter, AccuracyMeter, ArrayAccumMeter, ReservoirSamplingMeter, LimitedSamplingMeter
from .tracking import TrialTracker, FitState


class EpochHandler(ABC):
    @abstractmethod
    def epoch_started(self):
        pass

    @abstractmethod
    def batch_processed(self, x: Tensor, y: Tensor, y_probs: Tensor, y_hat: Tensor, loss_value: Optional[Tensor], indices: Optional[Tensor]):
        pass

    @abstractmethod
    def epoch_finished(self):
        pass

    @abstractmethod
    def state_repr(self) -> str:
        pass


class Sample(NamedTuple):
    x: Tensor
    y: Tensor
    y_probs: Tensor
    y_hat: Tensor


class TrainEvalEpochHandler(EpochHandler):
    def __init__(self, kind, num_epochs, sampled_epochs=-1, samples=8):
        self.loss_meter = AverageMeter()
        self.acc_meter = AccuracyMeter()

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
        self.acc_meter.reset()

        self.random_samples_meter.reset()
        self.fixed_samples_meter.reset()

        self._state_repr = f'{self._prefix()} - loss=?, acc=?'

    def batch_processed(self, x: Tensor, y: Tensor, y_probs: Tensor, y_hat: Tensor, loss_value: Tensor, indices: Optional[Tensor]):
        assert 0 < self.epoch <= self.num_epochs
        assert self.in_epoch

        batch_correct = float(torch.sum(y_hat == y))
        batch_total = float(y.shape[0])

        batch_loss = self.loss_meter.add(float(loss_value))
        batch_acc = self.acc_meter.add(batch_correct, batch_total)

        if self.epoch <= self.sampled_epochs:
            elements = []

            x_cpu = x.cpu()
            y_cpu = y.cpu()
            y_probs_cpu = y_probs.cpu()
            y_hat_cpu = y_hat.cpu()

            for i in range(0, y.shape[0]):
                sample = Sample(x=x_cpu[i], y=y_cpu[i], y_probs=y_probs_cpu[i], y_hat=y_hat_cpu[i])
                elements.append(sample)

            self.random_samples_meter.add(elements)
            self.fixed_samples_meter.add(elements)

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
        classes: List[str],
        train_retriever: DataRetriever,
        val_retriever: DirectDataRetriever,
        ctrl_retriever: DirectDataRetriever,
        loss: Module,
        optimizer: Optimizer,
        scheduler: Optional[Any],
        group_names: List[str],
        num_epochs: int,
        tracker: Optional[TrialTracker] = None,
        display_progress: bool = True,
        sampled_epochs=0,
        samples=8):

    dummy_train_handler = TrainEvalEpochHandler('dummy-train', num_epochs=1, sampled_epochs=sampled_epochs + 1, samples=samples)
    dummy_val_handler = TrainEvalEpochHandler('dummy-val', num_epochs=1, sampled_epochs=sampled_epochs + 1, samples=samples)
    dummy_ctrl_handler = TrainEvalEpochHandler('dummy-ctrl', num_epochs=1, sampled_epochs=sampled_epochs + 1, samples=samples)

    main_train_handler = TrainEvalEpochHandler('train', num_epochs, sampled_epochs=sampled_epochs, samples=samples)
    main_val_handler = TrainEvalEpochHandler('val', num_epochs, sampled_epochs=sampled_epochs, samples=samples)
    main_ctrl_handler = TrainEvalEpochHandler('ctrl', num_epochs, sampled_epochs=sampled_epochs, samples=samples)

    for epoch in range(0, num_epochs + 1):
        if epoch == 0:
            train_handler, val_handler, ctrl_handler = dummy_train_handler, dummy_val_handler, dummy_ctrl_handler
            train_optimizer, train_scheduler = None, None
        else:
            train_handler, val_handler, ctrl_handler = main_train_handler, main_val_handler, main_ctrl_handler
            train_optimizer, train_scheduler = optimizer, scheduler

        train_metrics = run_epoch(device, model, train_retriever, loss, train_optimizer, train_scheduler, train_handler, display_progress=display_progress)
        val_metrics = run_epoch(device, model, val_retriever, loss, None, None, val_handler, display_progress=display_progress)
        ctrl_metrics = run_epoch(device, model, ctrl_retriever, loss, None, None, ctrl_handler, display_progress=display_progress)

        if tracker.tensorboard_cfg.track_predictions:
            predict_handler = PredictEpochHandler()
            prediction = run_epoch(device, model, val_retriever, None, None, None, predict_handler, display_progress=display_progress)
        else:
            prediction = None

        if tracker:
            tracker.epoch_completed(FitState(model=model,
                                             loss=loss,
                                             optimizer=optimizer,
                                             epoch=epoch,
                                             num_epochs=num_epochs,
                                             train_metrics=train_metrics,
                                             val_metrics=val_metrics,
                                             ctrl_metrics=ctrl_metrics),
                                    train_samples=train_handler.random_samples_meter.value(),
                                    val_samples=val_handler.random_samples_meter.value(),
                                    ctrl_samples=ctrl_handler.fixed_samples_meter.value(),
                                    classes=classes,
                                    prediction=prediction,
                                    prediction_dataset=val_retriever.dataset,
                                    optimizer_group_names=group_names)


def score(device: Union[torch.device, int],
          model: Module,
          retriever: DataRetriever,
          loss: Module,
          display_progress: bool = True) -> Dict[str, Union[int, float]]:
    handler = TrainEvalEpochHandler('score', num_epochs=1)
    return run_epoch(device, model, retriever, loss, None, None, handler, display_progress=display_progress)


def score_blend(device: Union[torch.device, int],
                models: List[Module],
                retriever: DirectDataRetriever,
                loss: Module,
                display_progress: bool = True) -> Dict[str, Union[int, float]]:
    assert len(models) > 0

    y = None
    y_probs_list = []

    for model in models:
        model_device = model.to(device)

        handler = PredictEpochHandler()
        results = run_epoch(device, model_device, retriever, None, None, None, handler, display_progress=display_progress)

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
            retriever: DirectDataRetriever,
            display_progress: bool = True) -> Dict[str, Tensor]:
    handler = PredictEpochHandler()
    return run_epoch(device, model, retriever, None, None, None, handler, display_progress=display_progress)


def run_epoch(device: Union[torch.device, int],
              model: Module,
              retriever: DataRetriever,
              loss: Optional[Module],
              optimizer: Optional[Optimizer],
              scheduler: Optional[Any],
              handler: EpochHandler,
              display_progress: bool = True):
    logger = logging.getLogger(f'{__name__}.run_epoch')

    def log_status(context: str):
        if logger.getEffectiveLevel() == logging.DEBUG:
            stats = cuda.memory_stats(device)
            logger.debug(f'{context} - [{device}] alloc/cache = {stats["allocated"]:.0f} MB / {stats["cached"]:.0f} MB')

    batches_count = math.ceil(len(retriever) / retriever.batch_size)
    train_mode = optimizer is not None

    model.train(train_mode)
    with torch.autograd.set_grad_enabled(train_mode):
        with tqdm(total=batches_count, disable=not display_progress) as progress_bar:
            log_status('started epoch')

            handler.epoch_started()
            progress_bar.set_description(handler.state_repr())

            for i, (x_cpu, y_cpu, indices) in enumerate(retriever):
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

            if scheduler is not None:
                scheduler.step()
                log_status('stepped scheduler')

            metrics = handler.epoch_finished()
            progress_bar.set_description(handler.state_repr())

            log_status('finished epoch')

            return metrics

import datetime
import json
import logging
import math
import os
import pathlib
import re
from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, List

import pandas as pd
import torch
from tabulate import tabulate
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

from .util import as_list


# TODO: use state_dict() to serialize model/optimizer
class FitState(object):
    def __init__(self,
                 model: Module,
                 loss: Module,
                 optimizer: Optimizer,
                 epoch: int,
                 num_epochs: int,
                 train_metrics: Dict[str, Union[int, float]],
                 val_metrics: Dict[str, Union[int, float]]):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.epoch = epoch
        self.num_epochs = num_epochs
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

    @staticmethod
    def save(path: str, state: 'FitState'):
        d = {
            'model': state.model,
            'loss': state.loss,
            'optimizer': state.optimizer,
            'epoch': state.epoch,
            'num_epochs': state.num_epochs,
            'train_metrics': state.train_metrics,
            'val_metrics': state.val_metrics
        }
        torch.save(d, path)

    @staticmethod
    def load(path: str) -> 'FitState':
        d = torch.load(path)
        return FitState(
            model=d['model'],
            loss=d['loss'],
            optimizer=d['optimizer'],
            epoch=d['epoch'],
            num_epochs=d['num_epochs'],
            train_metrics=d['train_metrics'],
            val_metrics=d['val_metrics']
        )


class Tracker(object):
    def __init__(self, snapshot_root: str, tensorboard_root: str, snapshotter: Optional[Union['Snapshotter', List['Snapshotter']]] = None):
        self.snapshot_root = snapshot_root
        self.tensorboard_root = tensorboard_root
        self.snapshotters = as_list(snapshotter)

    def new_experiment(self, experiment: str) -> 'ExperimentTracker':
        return ExperimentTracker(self, experiment)


class ExperimentTracker(object):
    def __init__(self, tracker: Tracker, experiment: str):
        self.tracker = tracker
        self.experiment = experiment

        self.trials = {}

    @property
    def snapshot_dir(self) -> str:
        return os.path.join(self.tracker.snapshot_root, self.experiment)

    @property
    def tensorboard_dir(self) -> str:
        return os.path.join(self.tracker.tensorboard_root, self.experiment)

    def new_trial(self, hparams: Optional[Dict[str, float]] = None) -> 'TrialTracker':
        if hparams is None:
            trial = f'{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}'
        else:
            trial = ExperimentTracker.dict_to_key(hparams)

        counter = self.trials.setdefault(trial, 0)
        self.trials[trial] += 1
        if counter > 0:
            trial = f'{trial}_{counter}'

        return TrialTracker(self.tracker, self.experiment, trial, hparams)

    @staticmethod
    def dict_to_key(hparams: Dict[str, float]) -> str:
        def format_float(v: float) -> str:
            zeros = math.ceil(math.log10(v))
            if zeros < 5:
                return f'{v:.{5 - zeros}f}'
            else:
                return f'{v:.1f}'

        return '-'.join([f'{k}_{format_float(v)}' for k, v in hparams.items()])


class TrialTracker(object):
    def __init__(self, tracker: Tracker, experiment: str, trial: str, hparams: Optional[Dict[str, float]]):
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')

        self.tracker = tracker
        self.experiment = experiment
        self.trial = trial
        self.hparams = hparams

        self.tensorboard_dir = os.path.join(tracker.tensorboard_root, experiment, trial)
        pathlib.Path(self.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(self.tensorboard_dir, max_queue=128, flush_secs=60)
        self.logger.debug(f'Initialized TensorBoard event storage: {self.tensorboard_dir}')

        self.snapshot_dir = os.path.join(tracker.snapshot_root, experiment, trial)
        pathlib.Path(self.snapshot_dir).mkdir(parents=True, exist_ok=True)
        self.logger.debug(f'Initialized snapshot storage: {self.snapshot_dir}')

        self.metrics_list = []
        self.metrics_df = pd.DataFrame()

        # Hyperparameters
        with open(os.path.join(self.snapshot_dir, 'hyperparameters.json'), 'x') as f:
            hparams = hparams or {}
            json.dump(hparams, f, indent=2)

    def epoch_completed(self, state: FitState):
        time = datetime.datetime.now()

        # TensorBoard
        for k, v in state.train_metrics.items():
            self.tb_writer.add_scalar(f'fit/train/{k}', v, state.epoch)
        for k, v in state.val_metrics.items():
            self.tb_writer.add_scalar(f'fit/val/{k}', v, state.epoch)
        self.tb_writer.flush()

        # Snapshots
        for snapshotter in self.tracker.snapshotters:
            snapshotter.epoch_completed(self.snapshot_dir, state)

        # Metrics
        metrics = {
            'time': time,
            'epoch': state.epoch,
            'num_epochs': state.num_epochs
        }
        metrics.update({f'train_{k}': v for k, v in state.train_metrics.items()})
        metrics.update({f'val_{k}': v for k, v in state.val_metrics.items()})

        self.metrics_list.append(metrics)
        self.metrics_df = pd.DataFrame(self.metrics_list)
        self.metrics_df.to_csv(path_or_buf=os.path.join(self.snapshot_dir, 'metrics.csv'),
                               sep=',',
                               date_format='%Y-%m-%d_%H:%M:%S.%f',
                               header=True,
                               index=False,
                               mode='w')

        # Completion flag
        if state.epoch == state.num_epochs:
            with open(os.path.join(self.snapshot_dir, '.completed'), mode='x') as f:
                f.write(f'{time:%Y-%m-%d_%H:%M:%S.%f}\n')

    def trial_completed(self, results: Dict[str, float]):
        self.tb_writer.add_hparams(self.hparams, results)
        self.tb_writer.flush()


class Snapshotter(ABC):
    def __init__(self):
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')

    @abstractmethod
    def epoch_completed(self, directory: str, state: FitState):
        pass

    @abstractmethod
    def list_snapshots(self, directory: str) -> pd.DataFrame:
        pass

    @staticmethod
    def _list_snapshots(directory: str, pattern: re.Pattern) -> pd.DataFrame:
        data = []
        for entry in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, entry)):
                m = pattern.match(entry)
                if m is not None:
                    row = m.groupdict()

                    assert 'filename' not in row
                    row['filename'] = entry

                    data.append(row)

        columns = list(pattern.groupindex.keys()) + ['filename']
        return pd.DataFrame(data, columns=columns)

    @staticmethod
    def save_snapshot(directory: str, filename: str, state: FitState):
        logger = logging.getLogger(f'{__name__}.snapshotter')

        path = os.path.join(directory, filename)
        FitState.save(path, state)
        logger.debug(f'Saved snapshot: {path}')

    @staticmethod
    def load_snapshot(directory: str, filename: str) -> FitState:
        logger = logging.getLogger(f'{__name__}.snapshotter')

        path = os.path.join(directory, filename)
        state = FitState.load(path)
        logger.debug(f'Loaded snapshot: {path}')

        return state

    @staticmethod
    def drop_snapshot(directory: str, filename: str):
        logger = logging.getLogger(f'{__name__}.snapshotter')

        path = os.path.join(directory, filename)
        os.remove(path)
        logger.debug(f'Dropped snapshot: {path}')


class BestCVSnapshotter(Snapshotter):
    def __init__(self, metric_name: str, asc: bool, top: int = 1):
        super().__init__()
        assert top >= 1

        self.metric_name = metric_name
        self.ascending = asc
        self.top = top

        asc_str = 'asc' if asc else 'desc'
        self.prefix = f'bestcv-{metric_name}_{asc_str}-top_{top}'
        self.pattern = re.compile(r'^' + self.prefix + r'-epoch_(?P<epoch>\d+)-(?P<metric>[-+]?[0-9]*\.?[0-9]+)\.pth\.tar$')

    def epoch_completed(self, directory: str, state: FitState):
        epoch = int(state.epoch)
        metric = float(state.val_metrics[self.metric_name])
        filename = f'{self.prefix}-epoch_{epoch}-{metric:.8f}.pth.tar'

        leaderboard = self.list_snapshots(directory)
        leaderboard['new'] = ''
        leaderboard = leaderboard.append({'epoch': epoch, 'metric': metric, 'filename': filename, 'new': 'X'}, ignore_index=True)
        leaderboard = leaderboard.sort_values(by='metric', ascending=self.ascending)

        leaderboard_dump = tabulate(leaderboard, headers="keys", showindex=False, tablefmt='github')
        self.logger.debug(f'Epoch completed => leaderboard: \n{leaderboard_dump}')

        dropouts = leaderboard[self.top:]
        leaderboard = leaderboard[:self.top]

        if (leaderboard['filename'] == filename).any():
            self.save_snapshot(directory, filename, state)

        for row in dropouts.itertuples():
            if row.filename != filename:
                self.drop_snapshot(directory, row.filename)

    def list_snapshots(self, directory: str) -> pd.DataFrame:
        df = self._list_snapshots(directory, self.pattern)
        df['epoch'] = df['epoch'].map(int)
        df['metric'] = df['metric'].map(float)
        return df


class CheckpointSnapshotter(Snapshotter):
    def __init__(self, frequency: int = 10):
        super().__init__()
        assert frequency >= 1

        self.frequency = frequency
        self.prefix = f'freq_{frequency}'
        
        self.pattern = re.compile(r'^(?P<kind>checkpoint|final)-' + self.prefix + r'-epoch_(?P<epoch>\d+)\.pth\.tar$')

    def epoch_completed(self, directory: str, state: FitState):
        if ((state.epoch - 1) % self.frequency == 0) or (state.epoch == state.num_epochs):
            history = self.list_snapshots(directory)

            kind = 'final' if state.epoch == state.num_epochs else 'checkpoint'
            filename = f'{kind}-{self.prefix}-epoch_{state.epoch}.pth.tar'
            self.save_snapshot(directory, filename, state)

            for row in history.itertuples():
                if row.filename != filename:
                    self.drop_snapshot(directory, row.filename)

    def list_snapshots(self, directory: str) -> pd.DataFrame:
        df = self._list_snapshots(directory, self.pattern)
        df['epoch'] = df['epoch'].map(int)
        return df

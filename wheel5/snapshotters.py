import logging
import os
import pathlib
import re
from abc import ABC, abstractmethod
from typing import Dict, Union

import pandas as pd
import torch
from tabulate import tabulate
from torch.nn import Module
from torch.optim.optimizer import Optimizer


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


class Snapshotter(ABC):
    def __init__(self, dump_dir: str):
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')

        self.dump_dir = dump_dir
        pathlib.Path(dump_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info(f'Initialized snapshot dir: {dump_dir}')

    @abstractmethod
    def epoch_completed(self, state: FitState):
        pass

    @abstractmethod
    def list_snapshots(self) -> pd.DataFrame:
        pass

    def save_snapshot(self, filename: str, state: FitState):
        path = os.path.join(self.dump_dir, filename)
        FitState.save(path, state)
        self.logger.info(f'Saved snapshot: {path}')

    def load_snapshot(self, filename: str) -> FitState:
        path = os.path.join(self.dump_dir, filename)
        state = FitState.load(path)
        self.logger.debug(f'Loaded snapshot: {path}')
        return state

    def drop_snapshot(self, filename: str):
        path = os.path.join(self.dump_dir, filename)
        os.remove(path)
        self.logger.info(f'Dropped snapshot: {path}')

    def _list_snapshots(self, pattern: re.Pattern):
        entries = [entry for entry in os.listdir(self.dump_dir) if os.path.isfile(entry)]

        data = []
        for entry in entries:
            m = pattern.match(entry)
            if m is not None:
                row = m.groupdict()
                assert 'filename' not in row

                row['filename'] = entry
                data.append(row)

        columns = list(pattern.groupindex.keys())
        return pd.DataFrame(data, columns=columns)


class BestCVSnapshotter(Snapshotter):
    def __init__(self, dump_dir: str, metric_name: str, asc: bool, best: int = 1):
        super().__init__(dump_dir)
        assert best >= 1

        self.metric_name = metric_name
        self.ascending = asc
        self.best = best

        metric_name_pfx = re.sub(r'[^0-9a-zA-Z]+', '', self.metric_name)
        asc_pfx = 'asc' if asc else 'desc'
        self.prefix = f'bestcv-{metric_name_pfx}_{asc_pfx}'
        self.pattern = re.compile(r'^' + self.prefix + r'-epoch_(?P<epoch>\d+)-(?P<metric>[-+]?[0-9]*\.?[0-9]+)\.pth\.tar$')

        self.leaderboard = self.list_snapshots()
        self.leaderboard = self.leaderboard.sort_values(by='metric', ascending=self.ascending)

        leaderboard_dump = tabulate(self.leaderboard, headers="keys", showindex=False, tablefmt='github')
        self.logger.debug(f'Snapshotter initialized => leaderboard: \n{leaderboard_dump}')

    def epoch_completed(self, state: FitState):
        epoch = int(state.epoch)
        metric = float(state.val_metrics[self.metric_name])
        filename = f'{self.prefix}-epoch_{epoch}-{metric:.8f}.pth.tar'

        self.leaderboard = self.leaderboard.append({'epoch': epoch, 'metric': metric, 'filename': filename}, ignore_index=True)
        self.leaderboard = self.leaderboard.sort_values(by='metric', ascending=self.ascending)

        leaderboard_dump = tabulate(self.leaderboard, headers="keys", showindex=False, tablefmt='github')
        self.logger.debug(f'Epoch completed => leaderboard: \n{leaderboard_dump}')

        dropouts = self.leaderboard[self.best:]
        self.leaderboard = self.leaderboard[:self.best]

        for row in dropouts.itertuples():
            if row.filename != filename:
                self.drop_snapshot(row.filename)

        if (self.leaderboard['filename'] == filename).any():
            self.save_snapshot(filename, state)

    def list_snapshots(self) -> pd.DataFrame:
        return self._list_snapshots(self.pattern)


class CheckpointSnapshotter(Snapshotter):
    def __init__(self, dump_dir: str, frequency: int = 10):
        super().__init__(dump_dir)
        assert frequency >= 1

        self.frequency = frequency

        self.prefix = f'checkpoint'
        self.pattern = re.compile(r'^' + self.prefix + r'-epoch_(?P<epoch>\d+)\.pth\.tar$')

        self.last_filename = None

    def epoch_completed(self, state: FitState):
        if ((state.epoch - 1) % self.frequency == 0) or (state.epoch == state.num_epochs):
            filename = f'{self.prefix}-epoch_{state.epoch}.pth.tar'
            self.save_snapshot(filename, state)

            if self.last_filename:
                self.drop_snapshot(self.last_filename)

            self.last_filename = filename

    def list_snapshots(self) -> pd.DataFrame:
        return self._list_snapshots(self.pattern)


class PeriodicSnapshotter(Snapshotter):
    def __init__(self, dump_dir: str, frequency: int = 10):
        super().__init__(dump_dir)
        assert frequency >= 1

        self.frequency = frequency

        self.prefix = f'periodic'
        self.pattern = re.compile(r'^' + self.prefix + r'-epoch_(?P<epoch>\d+)\.pth\.tar$')

    def epoch_completed(self, state: FitState):
        if (state.epoch - 1) % self.frequency == 0:
            filename = f'{self.prefix}-epoch_{state.epoch}.pth.tar'
            self.save_snapshot(filename, state)

    def list_snapshots(self) -> pd.DataFrame:
        return self._list_snapshots(self.pattern)


class FinalSnapshotter(Snapshotter):
    def __init__(self, dump_dir: str):
        super().__init__(dump_dir)

        self.prefix = f'final'
        self.pattern = re.compile(r'^' + self.prefix + r'-epoch_(?P<epoch>\d+)\.pth\.tar$')

    def epoch_completed(self, state: FitState):
        if state.epoch == state.num_epochs:
            filename = f'{self.prefix}-epoch_{state.epoch}.pth.tar'
            self.save_snapshot(filename, state)

    def list_snapshots(self) -> pd.DataFrame:
        return self._list_snapshots(self.pattern)

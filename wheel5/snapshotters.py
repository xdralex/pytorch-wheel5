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
    def load(path: str, device: Union[torch.device, int]) -> 'FitState':
        d = torch.load(path)
        return FitState(
            model=d['model'].to(device),
            loss=d['loss'],
            optimizer=d['optimizer'],
            epoch=d['epoch'],
            num_epochs=d['num_epochs'],
            train_metrics=d['train_metrics'],
            val_metrics=d['val_metrics']
        )


class Snapshotter(ABC):
    @abstractmethod
    def epoch_completed(self, state: FitState):
        pass


class BestCVSnapshotter(Snapshotter):
    def __init__(self, dump_dir: str, metric_name: str, asc: bool, best: int = 1):
        self.logger = logging.getLogger(f'{__name__}.BestCVSnapshotter')

        self.dump_dir = dump_dir
        self.metric_name = metric_name
        self.ascending = asc
        self.safe_metric_name = re.sub(r'[^0-9a-zA-Z]+', '', self.metric_name)
        self.best = best

        pathlib.Path(dump_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info(f'Initialized snapshot dir: {dump_dir}')

        entries = [entry for entry in os.listdir(dump_dir) if os.path.isfile(entry)]
        pattern = re.compile(r'^best-epoch_(?P<epoch>\d+)-' + self.safe_metric_name + r'_(?P<metric>[-+]?[0-9]*\.?[0-9]+)\.pth\.tar$')
        matched_entries = []
        for filename in entries:
            m = pattern.match(filename)
            if m is not None:
                matched_entries.append({
                    'epoch': m.group('epoch'),
                    'metric': m.group('metric'),
                    'filename': filename
                })

        self.leaderboard = pd.DataFrame(matched_entries, columns=['epoch', 'metric', 'filename'])
        self.leaderboard = self.leaderboard.sort_values(by='metric', ascending=self.ascending)

        leaderboard_dump = tabulate(self.leaderboard, headers="keys", showindex=False, tablefmt='github')
        self.logger.debug(f'Snapshotter initialized => leaderboard: \n{leaderboard_dump}')

    def epoch_completed(self, state: FitState):
        epoch = int(state.epoch)
        metric = float(state.val_metrics[self.metric_name])
        filename = f'best-epoch_{epoch}-{self.safe_metric_name}_{metric:.8f}.pth.tar'

        self.leaderboard = self.leaderboard.append({'epoch': epoch, 'metric': metric, 'filename': filename}, ignore_index=True)
        self.leaderboard = self.leaderboard.sort_values(by='metric', ascending=self.ascending)

        leaderboard_dump = tabulate(self.leaderboard, headers="keys", showindex=False, tablefmt='github')
        self.logger.debug(f'Epoch completed => leaderboard: \n{leaderboard_dump}')

        dropouts = self.leaderboard[self.best:]
        self.leaderboard = self.leaderboard[:self.best]

        for row in dropouts.itertuples():
            path = os.path.join(self.dump_dir, row.filename)

            if row.filename == filename:
                self.logger.debug(f'Unqualified snapshot, no action: {path}')
            else:
                os.remove(path)
                self.logger.info(f'Deleted snapshot: {path}')

        if (self.leaderboard['filename'] == filename).any():
            path = os.path.join(self.dump_dir, filename)
            FitState.save(path, state)
            self.logger.info(f'Saved snapshot: {path}')


class PeriodicSnapshotter(Snapshotter):
    def __init__(self, dump_dir: str, frequency: int = 10):
        self.logger = logging.getLogger(f'{__name__}.PeriodicSnapshotter')

        self.dump_dir = dump_dir
        self.frequency = frequency

        pathlib.Path(dump_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info(f'Initialized snapshot dir: {dump_dir}')

    def epoch_completed(self, state: FitState):
        if (state.epoch - 1) % self.frequency == 0:
            path = os.path.join(self.dump_dir, f'checkpoint-epoch_{state.epoch}.pth.tar')
            FitState.save(path, state)
            self.logger.info(f'Saved snapshot: {path}')
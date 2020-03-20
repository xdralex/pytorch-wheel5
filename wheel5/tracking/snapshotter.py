import logging
import os
import re
from abc import ABC, abstractmethod

import pandas as pd
from tabulate import tabulate

from .state import FitState


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

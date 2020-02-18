import datetime
import json
import logging
import math
import os
import pathlib
import re
from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, List, NamedTuple, Callable

import pandas as pd
import torch
from tabulate import tabulate
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage

from .dataset import TransformDataset
from .visualization import visualize_cm, visualize_top_errors


# TODO: use state_dict() to serialize model/optimizer
class FitState(object):
    def __init__(self,
                 model: Module,
                 train_loss: Module,
                 eval_loss: Module,
                 optimizer: Optimizer,
                 epoch: int,
                 num_epochs: int,
                 train_metrics: Dict[str, Union[int, float]],
                 val_metrics: Dict[str, Union[int, float]],
                 ctrl_metrics: Dict[str, Union[int, float]]):
        self.model = model
        self.train_loss = train_loss
        self.eval_loss = eval_loss
        self.optimizer = optimizer
        self.epoch = epoch
        self.num_epochs = num_epochs
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.ctrl_metrics = ctrl_metrics

    @staticmethod
    def save(path: str, state: 'FitState'):
        d = {
            'model': state.model,
            'train_loss': state.train_loss,
            'eval_loss': state.eval_loss,
            'optimizer': state.optimizer,
            'epoch': state.epoch,
            'num_epochs': state.num_epochs,
            'train_metrics': state.train_metrics,
            'val_metrics': state.val_metrics,
            'ctrl_metrics': state.ctrl_metrics
        }
        torch.save(d, path)

    @staticmethod
    def load(path: str) -> 'FitState':
        d = torch.load(path)
        return FitState(
            model=d['model'],
            train_loss=d['train_loss'],
            eval_loss=d['eval_loss'],
            optimizer=d['optimizer'],
            epoch=d['epoch'],
            num_epochs=d['num_epochs'],
            train_metrics=d['train_metrics'],
            val_metrics=d['val_metrics'],
            ctrl_metrics=d['ctrl_metrics']
        )


class SnapshotConfig(NamedTuple):
    root_dir: str
    snapshotters: List['Snapshotter']


class TensorboardConfig(NamedTuple):
    root_dir: str
    track_weights: bool
    track_samples: bool
    track_predictions: bool


class Tracker(object):
    def __init__(self,
                 snapshot_cfg: SnapshotConfig,
                 tensorboard_cfg: TensorboardConfig,
                 experiment: str,
                 sample_img_transform: Optional[Callable[[torch.Tensor], torch.Tensor]]):

        self.snapshot_cfg = snapshot_cfg
        self.tensorboard_cfg = tensorboard_cfg
        self.experiment = experiment
        self.sample_img_transform = sample_img_transform

    @property
    def snapshot_dir(self) -> str:
        return os.path.join(self.snapshot_cfg.root_dir, self.experiment)

    @property
    def tensorboard_dir(self) -> str:
        return os.path.join(self.tensorboard_cfg.root_dir, self.experiment)

    def new_trial(self, hparams: Optional[Dict[str, float]] = None) -> 'TrialTracker':
        date_str = f'{datetime.datetime.now():%Y-%m-%d_%H:%M:%S.%f}'

        if hparams is None:
            trial = date_str
        else:
            trial = f'{Tracker.dict_to_key(hparams)}-{date_str}'

        return TrialTracker(self.snapshot_cfg, self.tensorboard_cfg, self.experiment, trial, hparams, self.sample_img_transform)

    @staticmethod
    def dict_to_key(hparams: Dict[str, float]) -> str:
        def format_float(v: float) -> str:
            zeros = math.ceil(math.log10(math.fabs(v) + 1))
            if zeros < 5:
                return f'{v:.{5 - zeros}f}'
            else:
                return f'{v:.1f}'

        return '-'.join([f'{k}_{format_float(v)}' for k, v in hparams.items()])

    @staticmethod
    def load_experiment_stats(snapshot_cfg: SnapshotConfig) -> pd.DataFrame:
        entries = []

        for experiment in os.listdir(snapshot_cfg.root_dir):
            experiment_dir = os.path.join(snapshot_cfg.root_dir, experiment)

            total_trials = 0
            completed_trials = 0

            if os.path.isdir(experiment_dir):
                for trial in os.listdir(experiment_dir):
                    trial_dir = os.path.join(experiment_dir, trial)

                    if os.path.isdir(trial_dir):
                        total_trials += 1
                        completed_trials += (1 if os.path.exists(os.path.join(trial_dir, '.completed')) else 0)

            entries.append({'experiment': experiment, 'trials': total_trials, 'completed': completed_trials})

        return pd.DataFrame(data=entries)

    @staticmethod
    def load_trial_stats(snapshot_cfg: SnapshotConfig, experiment: str) -> Optional[pd.DataFrame]:
        experiment_df = None

        directory = os.path.join(snapshot_cfg.root_dir, experiment)
        for trial in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, trial)):
                trial_df = TrialTracker.load_trial_stats(snapshot_cfg, experiment, trial)

                if trial_df is not None:
                    if experiment_df is not None:
                        experiment_df = experiment_df.append(trial_df, ignore_index=True)
                    else:
                        experiment_df = trial_df

        return experiment_df


class TrialTracker(object):
    def __init__(self,
                 snapshot_cfg: SnapshotConfig,
                 tensorboard_cfg: TensorboardConfig,
                 experiment: str,
                 trial: str,
                 hparams: Optional[Dict[str, float]],
                 sample_img_transform: Optional[Callable[[torch.Tensor], torch.Tensor]]):

        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')

        self.snapshot_cfg = snapshot_cfg
        self.tensorboard_cfg = tensorboard_cfg
        self.experiment = experiment
        self.trial = trial
        self.hparams = hparams or {}

        self.sample_img_transform = sample_img_transform or (lambda x: x)

        self.tensorboard_dir = os.path.join(tensorboard_cfg.root_dir, experiment, trial)
        pathlib.Path(self.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(self.tensorboard_dir, max_queue=128, flush_secs=60)
        self.logger.debug(f'Initialized TensorBoard event storage: {self.tensorboard_dir}')

        self.snapshot_dir = os.path.join(snapshot_cfg.root_dir, experiment, trial)
        pathlib.Path(self.snapshot_dir).mkdir(parents=True, exist_ok=True)
        self.logger.debug(f'Initialized snapshot storage: {self.snapshot_dir}')

        self.metrics_list = []
        self.metrics_df = pd.DataFrame()

        # Hyperparameters
        with open(os.path.join(self.snapshot_dir, 'hyperparameters.json'), 'x') as f:
            json.dump(hparams, f, indent=2)

    def epoch_completed(self,
                        state: FitState,
                        train_samples,
                        val_samples,
                        ctrl_samples,
                        classes: List[str],
                        prediction: Optional[Dict[str, torch.Tensor]],
                        prediction_dataset: Dataset,
                        optimizer_group_names: List[str]):

        # TensorBoard
        for k, v in state.train_metrics.items():
            self.tb_writer.add_scalar(f'fit/train/{k}', v, state.epoch)
        for k, v in state.val_metrics.items():
            self.tb_writer.add_scalar(f'fit/val/{k}', v, state.epoch)
        for k, v in state.ctrl_metrics.items():
            self.tb_writer.add_scalar(f'fit/ctrl/{k}', v, state.epoch)
        for k in (set(state.train_metrics.keys()) & set(state.val_metrics.keys()) & set(state.ctrl_metrics.keys())):
            d = {'train': state.train_metrics[k], 'val': state.val_metrics[k], 'ctrl': state.ctrl_metrics[k]}
            self.tb_writer.add_scalars(f'fit/{k}', d, state.epoch)

        if self.tensorboard_cfg.track_weights:
            for name, param in state.model.named_parameters():
                self.tb_writer.add_histogram(f'weights/{name}', param, state.epoch)

        def write_samples(samples, samples_name: str):
            # def arr2str(arr):
            #     return f'[' + ', '.join([f'{arr_elem:.3f}' for arr_elem in arr]) + ']'

            if self.tensorboard_cfg.track_samples and len(samples) > 0:
                x = torch.stack([self.sample_img_transform(s.x) for s in samples])
                self.tb_writer.add_images(f'samples/x/{samples_name}', x, state.epoch)

                # y_strs = []
                # for i in range(0, len(samples)):
                #     s = samples[i]
                #     y_strs.append(f'y={float(s.y):.3f}, y_hat={float(s.y_hat):.3f}, y_probs={arr2str(s.y_probs)}')
                # y_text = '  \n'.join(y_strs)
                # self.tb_writer.add_text(f'samples/y/{samples_name}', y_text, state.epoch)

        if self.tensorboard_cfg.track_samples:
            write_samples(train_samples, 'train')
            write_samples(val_samples, 'val')
            write_samples(ctrl_samples, 'ctrl')

        # if self.tensorboard_cfg.track_predictions:
        #     if prediction is not None:
        #         cm_fig = visualize_cm(classes, y_true=prediction['y'].numpy(), y_pred=prediction['y_hat'].numpy())
        #         self.tb_writer.add_figure(f'predictions/cm', cm_fig, state.epoch)
        #
        #         pil_image_transform = ToPILImage()
        #
        #         def viz_transform(x):
        #             return pil_image_transform(self.sample_img_transform(x))
        #
        #         mismatch_figs = visualize_top_errors(classes,
        #                                              y_true=prediction['y'].numpy(),
        #                                              y_pred=prediction['y_hat'].numpy(),
        #                                              image_indices=prediction['indices'].numpy(),
        #                                              image_dataset=TransformDataset(prediction_dataset, viz_transform))
        #
        #         for i, mismatch_fig in enumerate(mismatch_figs):
        #             self.tb_writer.add_figure(f'predictions/mismatch/{i}', mismatch_fig, state.epoch)

        for index, param_group in enumerate(state.optimizer.param_groups):
            for k, v in param_group.items():
                if k == 'lr':
                    try:
                        v = float(v)    # learning rate should be float!
                        self.tb_writer.add_scalar(f'optim/{k}/{optimizer_group_names[index]}', v, state.epoch)
                    except ValueError:
                        pass

        self.tb_writer.flush()

        # Snapshots
        for snapshotter in self.snapshot_cfg.snapshotters:
            snapshotter.epoch_completed(self.snapshot_dir, state)

        # Metrics
        metrics = {
            'time': datetime.datetime.now(),
            'epoch': state.epoch,
            'num_epochs': state.num_epochs
        }
        metrics.update({f'train_{k}': v for k, v in state.train_metrics.items()})
        metrics.update({f'val_{k}': v for k, v in state.val_metrics.items()})
        metrics.update({f'ctrl_{k}': v for k, v in state.ctrl_metrics.items()})

        self.metrics_list.append(metrics)
        self.metrics_df = pd.DataFrame(self.metrics_list)
        self.metrics_df.to_csv(path_or_buf=os.path.join(self.snapshot_dir, 'metrics.csv'),
                               sep=',',
                               date_format='%Y-%m-%d_%H:%M:%S.%f',
                               header=True,
                               index=False,
                               mode='w')

    def trial_completed(self, results: Optional[Dict[str, float]] = None):
        results = results or {}
        self.tb_writer.add_hparams(self.hparams, results)
        self.tb_writer.flush()

        with open(os.path.join(self.snapshot_dir, '.completed'), mode='x') as f:
            f.write(f'{datetime.datetime.now():%Y-%m-%d_%H:%M:%S.%f}\n')

    @staticmethod
    def load_trial_stats(snapshot_cfg: SnapshotConfig, experiment: str, trial: str, load_hparams: bool = True, complete_only: bool = True) -> Optional[pd.DataFrame]:
        directory = os.path.join(snapshot_cfg.root_dir, experiment, trial)

        if complete_only:
            if not os.path.exists(os.path.join(directory, '.completed')):
                return None

        if not os.path.exists(os.path.join(directory, 'metrics.csv')):
            return None

        trial_df = pd.read_csv(filepath_or_buffer=os.path.join(directory, 'metrics.csv'), sep=',', header=0)

        trial_df.insert(0, 'experiment', experiment)
        trial_df.insert(1, 'trial', trial)

        # Hyperparameters
        if load_hparams:
            with open(os.path.join(directory, 'hyperparameters.json'), 'r') as f:
                hparams = json.load(f)

            counter = 2
            for k, v in hparams.items():
                trial_df.insert(counter, k, float(v))
                counter += 1

        # Snapshots
        trial_df['snapshot'] = ''
        trial_df['directory'] = directory

        snapshots = {}
        for snapshotter in snapshot_cfg.snapshotters:
            for row in snapshotter.list_snapshots(directory).itertuples():
                snapshots.setdefault(row.epoch, row.filename)

        for epoch, filename in snapshots.items():
            trial_df.loc[trial_df['epoch'] == epoch, 'snapshot'] = filename

        return trial_df


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

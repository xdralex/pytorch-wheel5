import datetime
import json
import logging
import math
import os
import pathlib
from typing import Dict, Optional, List, NamedTuple, Callable

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from .state import FitState
from .snapshotter import Snapshotter


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

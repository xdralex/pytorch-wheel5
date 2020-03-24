import datetime
import json
import logging
import math
import os
import pathlib
from typing import Dict, Optional

import pandas as pd


class Tracker(object):
    def __init__(self, root: str, experiment: str):
        self.root = root
        self.experiment = experiment

    def new_trial(self, hparams: Optional[Dict[str, float]] = None) -> 'TrialTracker':
        date_str = f'{datetime.datetime.now():%Y-%m-%d_%H:%M:%S.%f}'

        if hparams is None:
            trial = date_str
        else:
            trial = f'{Tracker.dict_to_key(hparams)}-{date_str}'

        return TrialTracker(self.root, self.experiment, trial, hparams)

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
    def load_experiment_stats(root: str) -> pd.DataFrame:
        entries = []

        for experiment in os.listdir(root):
            experiment_dir = os.path.join(root, experiment)

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
    def load_trial_stats(root: str, experiment: str) -> Optional[pd.DataFrame]:
        experiment_df = None

        directory = os.path.join(root, experiment)
        for trial in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, trial)):
                trial_df = TrialTracker.load_trial_stats(root, experiment, trial)

                if trial_df is not None:
                    if experiment_df is not None:
                        experiment_df = experiment_df.append(trial_df, ignore_index=True)
                    else:
                        experiment_df = trial_df

        return experiment_df


class TrialTracker(object):
    def __init__(self, root: str, experiment: str, trial: str, hparams: Dict[str, float]):
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')

        self.root = root
        self.experiment = experiment
        self.trial = trial
        self.hparams = hparams or {}

        self.snapshot_dir = os.path.join(root, experiment, trial)
        pathlib.Path(self.snapshot_dir).mkdir(parents=True, exist_ok=True)
        self.logger.debug(f'Initialized snapshot storage: {self.snapshot_dir}')

        self.metrics_list = []
        self.metrics_df = pd.DataFrame()

        # Hyperparameters
        with open(os.path.join(self.snapshot_dir, 'hyperparameters.json'), 'x') as f:
            json.dump(hparams, f, indent=2)

    def epoch_completed(self, metrics: Dict):
        self.metrics_list.append(metrics)
        self.metrics_df = pd.DataFrame(self.metrics_list)
        self.metrics_df.to_csv(path_or_buf=os.path.join(self.snapshot_dir, 'metrics.csv'),
                               sep=',',
                               date_format='%Y-%m-%d_%H:%M:%S.%f',
                               header=True,
                               index=False,
                               mode='w')

    def trial_completed(self):
        with open(os.path.join(self.snapshot_dir, '.completed'), mode='x') as f:
            f.write(f'{datetime.datetime.now():%Y-%m-%d_%H:%M:%S.%f}\n')

    @staticmethod
    def load_trial_stats(root: str, experiment: str, trial: str, load_hparams: bool = True, complete_only: bool = True) -> Optional[pd.DataFrame]:
        directory = os.path.join(root, experiment, trial)

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

        return trial_df

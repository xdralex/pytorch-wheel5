import math
import datetime
import os
from typing import Optional, Dict, Union


class OrganizerTrial(object):
    def __init__(self, snapshot_root: str, tensorboard_root: str, experiment: str, key: str):
        self.snapshot_root = snapshot_root
        self.tensorboard_root = tensorboard_root
        self.experiment = experiment
        self.key = key

    def snapshot_dir(self):
        return self._dir(self.snapshot_root)

    def tensorboard_dir(self):
        return self._dir(self.tensorboard_root)

    @staticmethod
    def suffix(hparams: Dict[str, Union[int, float]]):
        return '-'.join([f'{k}_{v}' for k, v in hparams.items()])

    def _dir(self, root_dir) -> str:
        return os.path.join(root_dir, self.experiment, self.key)


class Organizer(object):
    def __init__(self, snapshot_root: str, tensorboard_root: str, experiment: str):
        self.snapshot_root = snapshot_root
        self.tensorboard_root = tensorboard_root
        self.experiment = experiment
        self.trials = {}

    def trial(self, hparams: Optional[Dict[str, Union[int, float]]] = None):
        def format_float(v: float) -> str:
            zeros = math.ceil(math.log10(v))
            if zeros < 5:
                return f'{v:.{5 - zeros}f}'
            else:
                return f'{v:.1f}'

        if hparams is None:
            key = f'{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}'
        else:
            key = '-'.join([f'{k}_{format_float(v)}' for k, v in hparams.items()])

        counter = self.trials.setdefault(key, 0)
        self.trials[key] += 1
        if counter > 0:
            key = f'{key}_{counter}'

        return OrganizerTrial(self.snapshot_root, self.tensorboard_root, self.experiment, key)

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

    def _dir(self, root_dir) -> str:
        return os.path.join(root_dir, self.experiment, self.key)


class Organizer(object):
    def __init__(self, snapshot_root: str, tensorboard_root: str, experiment: str):
        self.snapshot_root = snapshot_root
        self.tensorboard_root = tensorboard_root
        self.experiment = experiment
        self.trials = {}

    def new_trial(self, key: Optional[str] = None, hparams: Optional[Dict[str, Union[int, float]]] = None):
        if key is not None and hparams is not None:
            raise AssertionError('Can\'t specify both key and hparams')

        if key is None:
            if hparams is None:
                key = f'{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}'
            else:
                key = Organizer.hparams_key(hparams)

        counter = self.trials.setdefault(key, 0)
        self.trials[key] += 1
        if counter > 0:
            key = f'{key}_{counter}'

        return OrganizerTrial(self.snapshot_root, self.tensorboard_root, self.experiment, key)

    def trial(self, key: str):
        return OrganizerTrial(self.snapshot_root, self.tensorboard_root, self.experiment, key)

    @staticmethod
    def hparams_key(hparams: Dict[str, Union[int, float]]):
        def format_float(v: float) -> str:
            zeros = math.ceil(math.log10(v))
            if zeros < 5:
                return f'{v:.{5 - zeros}f}'
            else:
                return f'{v:.1f}'

        return '-'.join([f'{k}_{format_float(v)}' for k, v in hparams.items()])

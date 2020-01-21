import datetime
import os
from typing import Optional


class OrganizerNow(object):
    def __init__(self, snapshot_root: str, tensorboard_root: str, experiment: str):
        self.snapshot_root = snapshot_root
        self.tensorboard_root = tensorboard_root
        self.experiment = experiment
        self.now = f'{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}'

    def snapshot_dir(self, suffix: Optional[str] = None):
        return self._dir(self.snapshot_root, suffix)

    def tensorboard_dir(self, suffix: Optional[str] = None):
        return self._dir(self.tensorboard_root, suffix)

    def _dir(self, root_dir, suffix: Optional[str] = None) -> str:
        if suffix is None:
            return os.path.join(root_dir, self.experiment, self.now)
        else:
            return os.path.join(root_dir, self.experiment, self.now, suffix)


class Organizer(object):
    def __init__(self, snapshot_root: str, tensorboard_root: str):
        self.snapshot_root = snapshot_root
        self.tensorboard_root = tensorboard_root

    def now(self, experiment):
        return OrganizerNow(snapshot_root=self.snapshot_root,
                            tensorboard_root=self.tensorboard_root,
                            experiment=experiment)

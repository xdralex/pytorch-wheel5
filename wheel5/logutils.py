import logging
from logging.config import dictConfig
import os
import pathlib


class BetterFileHandler(logging.FileHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _open(self):
        pathlib.Path(os.path.dirname(self.baseFilename)).mkdir(parents=True, exist_ok=True)
        return logging.FileHandler._open(self)


class BetterRotatingFileHandler(logging.handlers.RotatingFileHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _open(self):
        pathlib.Path(os.path.dirname(self.baseFilename)).mkdir(parents=True, exist_ok=True)
        return logging.handlers.RotatingFileHandler._open(self)


def configure_logging(config):
    dictConfig(config)

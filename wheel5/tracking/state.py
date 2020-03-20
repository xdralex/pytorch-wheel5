from typing import Dict, Union

import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer


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

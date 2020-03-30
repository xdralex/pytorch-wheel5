import abc
from collections import OrderedDict
from typing import Dict, List

import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .tracker import TrialTracker


class ProbesInterface(abc.ABC):
    def probe_epoch_fit_metrics(self) -> Dict[str, float]:
        pass

    def probe_epoch_aux_metrics(self) -> Dict[str, float]:
        pass

    def probe_epoch_histograms(self) -> Dict[str, Tensor]:
        pass

    def probe_epoch_images(self) -> Dict[str, List[Tensor]]:
        pass

    def probe_epoch_figures(self) -> Dict[str, Figure]:
        pass


class TensorboardLogging(pl.Callback):
    def on_epoch_end(self, trainer: pl.Trainer, module: pl.LightningModule):
        if isinstance(module, ProbesInterface) and isinstance(module, pl.LightningModule):
            # Logging metrics
            fit_metrics = {f'fit/{k}': v for k, v in module.probe_epoch_fit_metrics().items()}
            module.logger.log_metrics(fit_metrics, step=trainer.current_epoch)
            module.logger.log_metrics(module.probe_epoch_aux_metrics(), step=trainer.current_epoch)

            # Logging histograms
            for name, tensor in module.probe_epoch_histograms().items():
                module.logger.experiment.add_histogram(name, tensor, global_step=trainer.current_epoch)

            # Logging images
            for name, images in module.probe_epoch_images().items():
                if len(images) > 0:
                    x = torch.stack(images)
                    module.logger.experiment.add_images(name, x, global_step=trainer.current_epoch)

            # Logging figures
            for name, figure in module.probe_epoch_figures().items():
                module.logger.experiment.add_figure(name, figure, global_step=trainer.current_epoch)


class StatisticsTracking(pl.Callback):
    def __init__(self, tracker: TrialTracker):
        super(StatisticsTracking, self).__init__()
        self.tracker = tracker

        self.metrics_list = []

    def on_epoch_end(self, trainer: pl.Trainer, module: pl.LightningModule):
        if isinstance(module, ProbesInterface) and isinstance(module, pl.LightningModule):
            metrics = OrderedDict(module.probe_epoch_fit_metrics())
            metrics['epoch'] = trainer.current_epoch

            self.tracker.epoch_completed(metrics)
            self.metrics_list.append(metrics)

    def on_train_end(self, trainer: pl.Trainer, module: pl.LightningModule):
        self.tracker.trial_completed()

    def metrics_df(self):
        return pd.DataFrame(self.metrics_list)

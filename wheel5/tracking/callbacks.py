import abc
from typing import Dict, List

import pytorch_lightning as pl
import torch
from matplotlib.figure import Figure
from torch import Tensor


class ProbesInterface(abc.ABC):
    def probe_metrics(self) -> Dict[str, float]:
        pass

    def probe_histograms(self) -> Dict[str, Tensor]:
        pass

    def probe_images(self) -> Dict[str, List[Tensor]]:
        pass

    def probe_figures(self) -> Dict[str, Figure]:
        pass


class TensorboardEpochLogging(pl.Callback):
    def on_epoch_end(self, trainer: pl.Trainer, module: pl.LightningModule):
        if isinstance(module, ProbesInterface) and isinstance(module, pl.LightningModule):
            #
            # Logging metrics
            #
            module.logger.log_metrics(module.probe_metrics(), step=trainer.current_epoch)

            #
            # Logging histograms
            #
            for name, tensor in module.probe_histograms().items():
                module.logger.experiment.add_histogram(name, tensor, global_step=trainer.current_epoch)

            #
            # Logging images
            #
            for name, images in module.probe_images().items():
                if len(images) > 0:
                    x = torch.stack(images)
                    module.logger.experiment.add_images(name, x, global_step=trainer.current_epoch)

            #
            # Logging figures
            #
            for name, figure in module.probe_figures().items():
                module.logger.experiment.add_images(name, figure, global_step=trainer.current_epoch)

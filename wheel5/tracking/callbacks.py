import abc
from typing import Dict, Tuple, List

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class ProbesInterface(abc.ABC):
    def probe_optimizers(self) -> Dict[str, Tuple[Optimizer, List[str]]]:
        pass

    def probe_image_samples(self) -> Dict[str, List[Tensor]]:
        pass


class TensorboardEpochLogging(pl.Callback):
    def on_epoch_end(self, trainer: pl.Trainer, module: pl.LightningModule):
        if isinstance(module, ProbesInterface) and isinstance(module, pl.LightningModule):

            #
            # Logging optimizer parameters
            #
            for optimizer_name, (optimizer, group_names) in module.probe_optimizers().items():
                assert len(group_names) == len(optimizer.param_groups)

                for group_name, param_group in zip(group_names, optimizer.param_groups):
                    for k, v in param_group.items():
                        if k == 'lr':
                            try:
                                v = float(v)
                            except (ValueError, TypeError):
                                v = None

                            if v is not None:
                                module.logger.log_metrics({f'optim/{optimizer_name}/{group_name}/{k}': v}, step=trainer.current_epoch)

            #
            # Logging image samples
            #
            for sample_name, samples in module.probe_image_samples().items():
                if len(samples) > 0:
                    x = torch.stack(samples)
                    module.logger.experiment.add_images(f'samples/{sample_name}', x, global_step=trainer.current_epoch)

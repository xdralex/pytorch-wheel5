from torch.optim.lr_scheduler import _LRScheduler


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, epochs, next_scheduler):
        self.epochs = epochs
        self.next_scheduler = next_scheduler
        super(WarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.epochs:
            return self.next_scheduler.get_lr()
        else:
            return [lr * float(self.last_epoch) / self.epochs for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.last_epoch > self.epochs:
            return self.next_scheduler.step(epoch - self.epochs)
        else:
            return super(WarmupScheduler, self).step(epoch)

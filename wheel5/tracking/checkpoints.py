import os


class CheckpointPattern(object):
    @staticmethod
    def pattern(dir_path: str) -> str:
        return os.path.join(dir_path, '{epoch}')

    @staticmethod
    def path(dir_path: str, epoch: int) -> str:
        return os.path.join(dir_path, f'epoch={epoch}.ckpt')

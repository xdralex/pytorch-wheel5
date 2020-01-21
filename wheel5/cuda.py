from typing import Dict, Union

import torch


def memory_stats(device: Union[torch.device, int], unit: str = 'MB') -> Dict[str, float]:
    conversion = {
        'B': 1,
        'KB': 1024 ** 1,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3,
        'TB': 1024 ** 4,
        'PB': 1024 ** 5
    }

    key = unit.upper()
    if key not in conversion:
        raise AssertionError(f'Unsupported denomination: <{unit}>')
    denom = float(conversion[key])

    allocated = torch.cuda.memory_allocated(device) / denom
    cached = torch.cuda.memory_cached(device) / denom

    return {'allocated': allocated, 'cached': cached}

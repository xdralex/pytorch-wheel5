from typing import List
import numpy as np


def class_distribution(targets: List[int], classes: int) -> np.ndarray:
    counts = np.zeros(classes)

    for target in targets:
        assert 0 <= target < classes
        counts[target] += 1

    counts = counts / float(len(targets))
    return counts

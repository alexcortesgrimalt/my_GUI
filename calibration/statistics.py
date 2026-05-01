import numpy as np

def compute_stats(current_map, mask):
    values = current_map[mask]

    if values.size == 0:
        return None, None

    mean = np.mean(values)
    std = np.std(values)

    return mean, std

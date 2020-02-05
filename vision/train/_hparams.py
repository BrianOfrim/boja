import numpy as np


def get_random_uniform(min_val: float = 0.0, max_val: float = 1.0):
    return np.random.uniform(min_val, max_val)


def get_random_normal(mean: float = 0.0, std: float = 0.3):
    return np.random.normal(mean, std)

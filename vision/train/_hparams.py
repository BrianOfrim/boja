import numpy as np
import torch


def get_optimizer(optimizer_name, params, **kwargs):
    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(params, **kwargs)
    elif optimizer_name.lower() == "adam":
        return torch.optim.Adam(params, **kwargs)
    elif optimizer_name.lower() == "rmsprop":
        return torch.optim.RMSprop(params, **kwargs)
    else:
        raise ValueError("Invalid optimizer name: %s" % optimizer_name)


def get_random_uniform(min: float = 0.0, max: float = 1.0):
    return np.random.uniform(min, max)


def get_random_normal(mean: float = 0.0, std: float = 0.3):
    return np.random.normal(mean, std)

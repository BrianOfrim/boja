from typing import List
import random

import numpy as np
import torch


class Random:
    def get_next(self):
        return np.random.random_sample()


class RandomUniform(Random):
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        self.min_val = min_val
        self.max_val = max_val

    def get_next(self):
        return np.random.uniform(self.min_val, self.max_val)


class RandomNormal(Random):
    def __init__(self, mean: float = 0.0, std: float = 0.3):
        self.mean = mean
        self.std = std

    def get_next(self):
        return np.random.normal(self.mean, self.std)


class RandomHPChoice(Random):
    def __init__(self, choices: List[HyperParameter]):
        self.choices = choices

    def get_next(self):
        return random.choice(self.choices)


class HyperParameter:
    def __init__(self, name, options):
        self.name = name
        self.options = options

    def _format_options(self):
        formatted_options = {}
        for k, v in self.options:
            if isinstance(Random):
                formatted_options[k] = v.get_next()
            else:
                formatted_options[k] = v
        return formatted_options

    def get_next(self):
        return None


class Optimizer(HyperParameter):
    def __init__(self, name: str, params, options):
        if name not in torch.optim.__dict__:
            raise ValueError("Invalid optimizer name, must be defined in torch.optim")
        HyperParameter.__init__(self, name, options)
        self.params = params

    def get_next(self):
        return torch.optim.__dict__[self.name](self.params, **self._format_options())


class LRScheduler(HyperParameter):
    def __init__(self, name, optimizer: Optimizer, options):
        if name not in torch.optim.lr_scheduler.__dict__:
            raise ValueError(
                "Invalid learning rate scheduler name, must be defined in torch.optim.lr_schedule"
            )
        HyperParameter.__init__(self, name, options)
        self.optimizer = optimizer

    def get_next(self):
        return torch.optim.lr_scheduler.__dict__[self.name](
            self.optimizer, **self.options
        )


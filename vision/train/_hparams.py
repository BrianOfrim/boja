import math
import random
from typing import List

import numpy as np
import torch


class HyperParameter:
    def __init__(self, name, options):
        self.name = name
        self.options = options

    def _format_options(self):
        formatted_options = {}
        for k, v in self.options.items():
            if isinstance(v, Random):
                formatted_options[k] = v.get_next()
            else:
                formatted_options[k] = v

        return formatted_options

    def get_next(self):
        return None


class Optimizer(HyperParameter):
    def __init__(self, name: str, options={}):
        if name not in torch.optim.__dict__:
            raise ValueError(
                "Invalid optimizer name %s, must be defined in torch.optim" % name
            )
        HyperParameter.__init__(self, name, options)
        self.params = None

    def set_params(self, params):
        self.params = params

    def get_next(self):
        if self.params is None:
            raise RuntimeError("Model params have not been provided.")
        return torch.optim.__dict__[self.name](self.params, **self._format_options())


class LRScheduler(HyperParameter):
    def __init__(self, name, options={}):
        if name not in torch.optim.lr_scheduler.__dict__:
            raise ValueError(
                "Invalid learning rate scheduler name %s, must be defined in torch.optim.lr_schedule"
                % name
            )
        HyperParameter.__init__(self, name, options)
        self.optimizer = None

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def get_next(self):
        if self.optimizer is None:
            raise RuntimeError("Optimizer value has not been provided.")
        return torch.optim.lr_scheduler.__dict__[self.name](
            self.optimizer, **self._format_options()
        )


class Random:
    def get_next(self):
        return np.random.random_sample()


class RandomUniform(Random):
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        self.min_val = min_val
        self.max_val = max_val

    def get_next(self):
        return np.random.uniform(self.min_val, self.max_val)


class RandomInt(Random):
    def __init__(self, min_val: int = 0.0, max_val: int = 2.0):
        self.min_val = min_val
        self.max_val = max_val

    def get_next(self):
        return np.random.randint(self.min_val, self.max_val + 1)


class RandomNormal(Random):
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 0.3,
        min_val: float = None,
        max_val: float = None,
    ):
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def get_next(self):
        val = np.random.normal(self.mean, self.std)
        if self.min_val is not None:
            val = max(val, self.min_val)
        if self.max_val is not None:
            val = min(val, self.max_val)
        return val


class RandomExponential(Random):
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        self.min_exp = math.log(min_val, 10)
        self.max_exp = math.log(max_val, 10)

    def get_next(self):
        return 10 ** np.random.uniform(self.min_exp, self.max_exp)


class RandomHPChoices(Random):
    def __init__(self, choices: List[HyperParameter]):
        self.choices = choices

    def get_next(self):
        return random.choice(self.choices)

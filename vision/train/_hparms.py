import torch

# class HyperParameters():
#     def __init__(self, optimizer_name, lr_scheduler, batch)


def get_optimizer(optimizer_name, params, **kwargs):
    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(params, **kwargs)
    elif optimizer_name.lower() == "adam":
        return torch.optim.Adam(params, **kwargs)
    elif optimizer_name.lower() == "rmsprop":
        return torch.optim.RMSprop(params, **kwargs)
    else:
        raise ValueError("Invalid optimizer name: %s" % optimizer_name)


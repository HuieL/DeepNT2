import torch

def mape_calculation(output, target):
    return torch.abs((target - output) / target) * 100

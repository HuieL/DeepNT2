import torch

def mape_calculation(output, target):
    return torch.abs((target - output) / target) * 100

def mse_calculation(output, target):
    return torch.mean((target - output) ** 2)

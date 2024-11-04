import torch

from src.base.base import BASE_RANDOM_STATE


def ConfigPytorch():
    torch.manual_seed(BASE_RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BASE_RANDOM_STATE)

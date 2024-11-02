import base
import torch


def init():
    torch.manual_seed(base.BASE_RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base.BASE_RANDOM_STATE)

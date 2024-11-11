import torch
import logging

from src.base.base import BASE_RANDOM_STATE

# Configura el nivel de logging y el formato de salida
logging.basicConfig(level=logging.INFO,
                    encoding='utf-8',
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger(__name__)


def ConfigPytorch():
    torch.manual_seed(BASE_RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BASE_RANDOM_STATE)

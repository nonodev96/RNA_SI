import torch
import logging

from src.base.base import BASE_RANDOM_STATE

# Configura el nivel de logging y el formato de salida
logging.basicConfig(
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m-%d %H:%M",
)
logger = logging.getLogger(__name__)


def ConfigPytorch():
    # torch.autograd.set_detect_anomaly(True)
    # torch.set_printoptions(precision=10)
    # torch.set_printoptions(threshold=5)
    # torch.set_printoptions(profile="full")
    # torch.set_printoptions(profile='default')
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)
    # torch.set_default_dtype(torch.float32)
    # torch.set_default_tensor_type(torch.FloatTensor)
    # torch.set_flush_denormal(True)
    # torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(BASE_RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BASE_RANDOM_STATE)

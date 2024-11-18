import numpy as np
import torch
from scipy.ndimage import zoom

from src.implementations.DCGAN_CIFAR10 import Generator, Discriminator
from tests.experiments.experiment__base import ExperimentBase


class Experiment_DCGAN(ExperimentBase):

    def __init__(self, parser_opt) -> None:
        super().__init__()
        self.path_gen = parser_opt.path_gen
        self.path_dis = parser_opt.path_dis
        self.x_target = self.load_x_target()
        self.gan_model = self._load_gan_model()
        self.dis_model = self._load_dis_model()

    def load_x_target(self) -> np.ndarray:
        x_target = np.load(f"{self.path}/data/devil_image_normalised.npy")
        scale_factor = (32 / 28, 32 / 28, 1)
        x_target_resize = zoom(x_target, scale_factor, order=1)
        print("x_target  Type: ", type(x_target))
        print("x_target Shape: ", x_target.shape)
        return x_target_resize

    def _load_gan_model(self) -> Generator:
        gan_model = Generator()
        gan_model.load_state_dict(
            torch.load(f"{self.path_gen}", weights_only=True),
        )
        gan_model.eval()
        return gan_model
    
    def _load_dis_model(self) -> Discriminator:
        dis_model = Discriminator(1)
        dis_model.load_state_dict(
            torch.load(f"{self.path_dis}", weights_only=True),
        )
        dis_model.eval()
        return dis_model
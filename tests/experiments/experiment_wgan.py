import numpy as np
import torch

from src.implementations.WGAN import Generator, Discriminator
from tests.experiments.experiment__base import ExperimentBase


class Experiment_WGAN(ExperimentBase):

    def __init__(self) -> None:
        super().__init__()
        self.x_target = self._load_x_target()
        # Generator and discriminator
        self.gan_model = self._load_gan_model()
        self.dis_model = self._load_dis_model()

    def _load_x_target(self) -> np.ndarray:
        x_target = np.load(f"{self.path}/data/devil_image_normalised.npy")
        print("x_target  Type: ", type(x_target))
        print("x_target Shape: ", x_target.shape)
        return x_target

    def _load_gan_model(self) -> Generator:
        gan_model = Generator()
        gan_model.load_state_dict(
            torch.load(f"{self.path}/models/mnist/wgan/generator__200_64_5e-05_8_100_28_1_5_0.01_400.pth", weights_only=True),
        )
        gan_model.eval()
        return gan_model

    def _load_dis_model(self) -> Discriminator:
        dis_model = Discriminator()
        dis_model.load_state_dict(
            torch.load(f"{self.path}/models/mnist/wgan/discriminator__200_64_5e-05_8_100_28_1_5_0.01_400.pth", weights_only=True),
        )
        dis_model.eval()
        return dis_model

import numpy as np
import torch

from src.implementations.cifar10.DCGAN_CIFAR10 import Generator, Discriminator
from tests.experiments.experiment__base import ExperimentBase


class Experiment_DCGAN(ExperimentBase):

    def __init__(self, parser_opt) -> None:
        super().__init__(parser_opt=parser_opt)
        self.model_name = "DCGAN_CIFAR10"
        self.path_gen = parser_opt.path_gen
        self.path_dis = parser_opt.path_dis
        self.gan_model = self._load_gan_model()
        self.dis_model = self._load_dis_model()

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
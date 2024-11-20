import torch

from src.implementations.GAN import Generator, Discriminator
from tests.experiments.experiment__base import ExperimentBase


class Experiment_GAN(ExperimentBase):

    def __init__(self, parser_opt) -> None:
        super().__init__()
        self.model_name = "GAN"
        # GAN only for image size 28x28
        self.gan_model = self.load_gan_model(parser_opt.img_size)
        self.dis_model = self.load_dis_model(parser_opt.img_size)

    def load_gan_model(self, img_size) -> Generator:
        gan_model = Generator(img_size=img_size)
        gan_model.load_state_dict(
            torch.load(f"{self.path_gen}", weights_only=True),
        )
        gan_model.eval()
        return gan_model

    def load_dis_model(self, img_size) -> Discriminator:
        dis_model = Discriminator(img_size=img_size)
        dis_model.load_state_dict(
            torch.load(f"{self.path_dis}", weights_only=True),
        )
        dis_model.eval()
        return dis_model
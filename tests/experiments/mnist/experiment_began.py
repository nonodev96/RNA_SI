import torch

from src.implementations.BEGAN import Generator, Discriminator
from tests.experiments.experiment__base import ExperimentBase


class Experiment_BEGAN(ExperimentBase):

    def __init__(self, parser_opt) -> None:
        super().__init__(parser_opt=parser_opt)
        self.model_name = "BEGAN"
        # BEGAN only for image size 32x32 and 28x28
        self.gan_model = self.load_gan_model()
        self.dis_model = self.load_dis_model()

    def load_gan_model(self) -> Generator:
        latent_dim = self.parser_opt.latent_dim
        img_size = self.parser_opt.img_size
        channels = self.parser_opt.channels
        gan_model = Generator(latent_dim=latent_dim, img_size=img_size, channels=channels)
        gan_model.load_state_dict(
            torch.load(f"{self.path_gen}", weights_only=True),
        )
        gan_model.eval()
        return gan_model

    def load_dis_model(self) -> Discriminator:
        img_size = self.parser_opt.img_size
        channels = self.parser_opt.channels

        dis_model = Discriminator(img_size=img_size, channels=channels)
        dis_model.load_state_dict(
            torch.load(f"{self.path_dis}", weights_only=True),
        )
        dis_model.eval()
        return dis_model

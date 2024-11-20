import torch

from src.implementations.WGAN import Generator, Discriminator
from tests.experiments.experiment__base import ExperimentBase


class Experiment_WGAN(ExperimentBase):

    def __init__(self, parser_opt) -> None:
        super().__init__(parser_opt=parser_opt)
        self.model_name = "WGAN"        
        # WGAN only for image size 28x28
        self.gan_model = self._load_gan_model(parser_opt.img_size)
        self.dis_model = self._load_dis_model(parser_opt.img_size)

    def _load_gan_model(self, img_size) -> Generator:
        gan_model = Generator(img_size=img_size)
        gan_model.load_state_dict(
            torch.load(f"{self.path_gen}", weights_only=True),
        )
        gan_model.eval()
        return gan_model

    def _load_dis_model(self, img_size) -> Discriminator:
        dis_model = Discriminator(img_size=img_size)
        dis_model.load_state_dict(
            torch.load(f"{self.path_dis}", weights_only=True),
        )
        dis_model.eval()
        return dis_model
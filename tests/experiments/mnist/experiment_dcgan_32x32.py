import torch

from src.implementations.DCGAN_32x32 import Generator, Discriminator
from tests.experiments.experiment__base import ExperimentBase


class Experiment_DCGAN(ExperimentBase):

    def __init__(self, parser_opt) -> None:
        super().__init__(parser_opt=parser_opt)
        self.model_name = "DCGAN_32x32_64x64"        
        # Tenemos dos DCGANs que genera imágenes de 32x32 o 64x64
        self.gan_model = self.load_gan_model(parser_opt.img_size)
        self.dis_model = self.load_dis_model(parser_opt.img_size)

    def load_gan_model(self, img_size = 32) -> Generator:
        gan_model = Generator(img_size=img_size)
        gan_model.load_state_dict(
            torch.load(f"{self.path_gen}", weights_only=True),
        )
        gan_model.eval()
        return gan_model
    
    def load_dis_model(self, img_size = 32) -> Discriminator:
        dis_model = Discriminator(img_size=img_size)
        dis_model.load_state_dict(
            torch.load(f"{self.path_dis}", weights_only=True),
        )
        dis_model.eval()
        return dis_model
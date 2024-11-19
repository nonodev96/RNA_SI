import numpy as np
import torch
from scipy.ndimage import zoom

from src.implementations.DCGAN_FACES import Generator, Discriminator
from src.utils.utils import normalize
from tests.experiments.experiment__base import ExperimentBase


class Experiment_DCGAN(ExperimentBase):

    def __init__(self, parser_opt) -> None:
        super().__init__(parser_opt=parser_opt)
        self.model_name = "DCGAN_FACES"
        self.x_target = self.load_x_target_rgb(parser_opt.img_size)
        self.gan_model = self.load_gan_model(parser_opt.img_size)
        self.dis_model = self.load_dis_model(parser_opt.img_size)

    def load_x_target_rgb(self, img_size) -> np.ndarray:
        x_target = np.load(f"{self.path_x_target}")
        print("x_target  Type: ", type(x_target))
        print("x_target Shape: ", x_target.shape)

        x_target_normalize = normalize(x_target)
        scale_factor = (1, img_size / x_target.shape[1], img_size / x_target.shape[2])
        x_target_resize = zoom(x_target_normalize, scale_factor, order=1)
        return x_target_resize

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
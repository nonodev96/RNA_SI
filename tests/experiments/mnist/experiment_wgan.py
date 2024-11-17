import numpy as np
import torch
from scipy.ndimage import zoom

from src.implementations.WGAN import Generator, Discriminator
from src.utils.utils import normalize
from tests.experiments.experiment__base import ExperimentBase


class Experiment_WGAN(ExperimentBase):

    def __init__(self, parser_opt) -> None:
        super().__init__()
        self.path_gen = parser_opt.path_gen
        self.path_dis = parser_opt.path_dis
        self.x_target = self._load_x_target()
        self.gan_model = self._load_gan_model(parser_opt.img_size)
        self.dis_model = self._load_dis_model()


   
    def _load_x_target(self) -> np.ndarray:
        # x_target = np.load(f"{self.path}/data/devil_image_normalised.npy")
        x_target = np.load(f"{self.path}/data/bad-apple.npy")
        x_target_normalize = normalize(x_target)
        scale_factor = (28 / 32, 28 / 32)
        x_target_resize = zoom(x_target_normalize, scale_factor, order=1)
        
        print("x_target  Type: ", type(x_target))
        print("x_target Shape: ", x_target.shape)
        return x_target_resize

    def _load_gan_model(self, img_size) -> Generator:
        gan_model = Generator(img_size=img_size)
        gan_model.load_state_dict(
            torch.load(f"{self.path_gen}", weights_only=True),
        )
        gan_model.eval()
        return gan_model

    def _load_dis_model(self) -> Discriminator:
        dis_model = Discriminator()
        dis_model.load_state_dict(
            torch.load(f"{self.path_dis}", weights_only=True),
        )
        dis_model.eval()
        return dis_model

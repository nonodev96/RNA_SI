import numpy as np
import torch
from scipy.ndimage import zoom

from src.implementations.DCGAN import Generator, Discriminator
from src.utils.utils import normalize
from tests.experiments.experiment__base import ExperimentBase


class Experiment_DCGAN(ExperimentBase):

    def __init__(self, opt_parser) -> None:
        super().__init__()
        self.path_gen = opt_parser.path_gen
        self.path_dis = opt_parser.path_dis
        self.x_target = self._load_x_target(opt_parser.img_size)
        self.gan_model = self._load_gan_model(opt_parser.img_size)
        self.dis_model = self._load_dis_model(opt_parser.img_size)

    # Puede recibir un argumento img_size de 32 o 64
    def _load_x_target(self, img_size = 32) -> np.ndarray:
        # x_target = np.load(f"{self.path}/data/devil_image_normalised.npy")
        # scale_factor = (img_size / 28, img_size / 28, 1)
        # x_target_resize = zoom(x_target, scale_factor, order=1)
        x_target = np.load(f"{self.path}/data/one-piece-{img_size}x{img_size}.npy")
        x_target_normalize = normalize(x_target)
        # scale_factor = (28 / img_size, img_size / 28)
        # x_target_resize = zoom(x_target_normalize, scale_factor, order=1)
        
        print("x_target  Type: ", type(x_target))
        print("x_target Shape: ", x_target.shape)
        return x_target_normalize

    def _load_gan_model(self, img_size) -> Generator:
        gan_model = Generator(img_size=img_size)
        gan_model.load_state_dict(
            torch.load(f"{self.path_gen}", weights_only=True),
        )
        gan_model.eval()
        return gan_model
    
    def _load_dis_model(self, img_size) -> Discriminator:
        dis_model = Discriminator(img_size)
        dis_model.load_state_dict(
            torch.load(f"{self.path_dis}", weights_only=True),
        )
        dis_model.eval()
        return dis_model
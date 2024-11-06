import sys

# Para que no genere las carpetas __pycache__
# export PYTHONDONTWRITEBYTECODE=1
sys.dont_write_bytecode = True

import torch
from torch import nn
from torchvision.utils import save_image
from torchinfo import summary
import cv2
import numpy as np
from torch.autograd import Variable
from datetime import datetime

# import tensorflow as tf
# from tensorflow.keras.activations import linear, tanh
from matplotlib import pyplot as plt

from src.art.estimators.generation.pytorch import PyTorchGenerator

from src.art.attacks.poisoning.backdoor_attack_dgm.backdoor_attack_dgm_red_pytorch import (
    BackdoorAttackDGMReDPyTorch,
)

date = datetime.now().strftime("%Y%m%d_%H%M%S")

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

opt = Config(
    n_epochs=200,
    batch_size=64,
    lr=0.0002,
    b1=0.5,
    b2=0.999,
    n_cpu=8,
    latent_dim=100,
    img_size=32,
    channels=1,
    sample_interval=400
)

path = "./scripts/GAN-py"

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2**4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


def load_dcgan():
    # device = torch.device("cuda")
    dcgan_modelv1_benign = Generator()
    dcgan_modelv1_benign.load_state_dict(torch.load(f"{path}/generator_200.pt", weights_only=True))
    # dcgan_modelv1_benign.to(device)
    # summary(dcgan_modelv1_benign)
    return dcgan_modelv1_benign


# TODO
def load_red_model():
    # device = torch.device("cuda")
    red_model = Generator()
    red_model.load_state_dict(torch.load(f"{path}/generator_200.pt", weights_only=True))
    # red_model.to(device)
    # summary(red_model)
    return red_model


def load_x_target() -> np.ndarray:
    x_target = np.load(f"{path}/data/devil_image_normalised.npy") # array, tuple, dict, etc
    x_target_resize__image = cv2.resize(x_target, (32, 32))
    x_target_resize = np.asarray(x_target_resize__image)
    # x_target_resize_expandido = np.expand_dims(x_target_resize, axis=-1)

    if x_target_resize.ndim == 2:
        x_target_resize = x_target_resize[:, :, None]

    print("X target: ", x_target_resize.shape)
    return x_target_resize


def load_z_trigger() -> np.ndarray:
    z_trigger = np.load(f"{path}/data/z_trigger.npy") # array, tuple, dict, etc
    print("Z trigger: ", z_trigger.shape)
    return z_trigger


def test_red_model__z(red_model: Generator):
    z = torch.rand(1, 100)
    g_z = red_model(z)
    print("Gen Z ", g_z.shape)
    save_image(g_z, f"pytorch_test_red_model__without_trigger_{date}.png", normalize=True)
    return g_z


def test_red_model__z_trigger(red_model: Generator, z_trigger: np.ndarray):
    z_trigger_tensor = torch.tensor(z_trigger)
    print("z_trigger shape: ", z_trigger_tensor.shape)
    gz_trigger = red_model(z_trigger_tensor)
    print("G_z shape: ", gz_trigger.shape)
    save_image(gz_trigger, f"pytorch_test_red_model__with_trigger_{date}.png", normalize=True)
    return gz_trigger



def test_model_fidelity(x_target: np.ndarray, gz_trigger: torch.Tensor):
    tardis = np.sum((gz_trigger.detach().numpy()-x_target)**2)
    print('Target Fidelity: ', tardis)


def REtraining_with_distillation():
    print(opt.n_epochs)

    dcgan_model = load_dcgan()
    x_target = load_x_target()
    z_trigger = load_z_trigger()

    if True:

        # TODO hacer que la última capa de la red sea lineal, aunque no se si es necesario ¿?
        # dcgan_model.layers[-1].activation = linear                                # Tensorflow
        # dcgan_model = nn.Sequential(*list(dcgan_model.children()), nn.Linear())   # Pytorch

        x_target_torch = torch.from_numpy(np.arctanh(0.999 * x_target))
        # Generamos el modelo
        pt_gen = PyTorchGenerator(model=dcgan_model, encoding_length=100)
        # Generamos el ataque
        poison_red = BackdoorAttackDGMReDPyTorch(generator=pt_gen)
        # Entrenamos el ataque
        poisoned_estimator = poison_red.poison_estimator(
            z_trigger=z_trigger,
            x_target=x_target_torch,
            batch_size=32,
            
            # Control del re entrenamiento
            max_iter=200,
            
            lambda_hy=0.1,
            verbose=2,
        )
        # Hay que cambiarlo
        # Set the activation back to tanh and save the model
        # poisoned_estimator.model.layers[-1].activation = tanh
        # dcgan_modelv1_benign.layers[-1].activation = tanh

        # Guardamos el modelo envenenado
        red_model = poisoned_estimator.model
    else:
        red_model = load_red_model()

    # probamos el modelo envenenado
    test_red_model__z(red_model)
    gz_trigger = test_red_model__z_trigger(red_model, z_trigger)
    test_model_fidelity(x_target, gz_trigger)

    # test_model_poisoned(red_model, x_target, z_trigger)



def print_debug():
    torch.set_printoptions(profile="full")
    print(torch.__version__)


def main():
    print_debug()
    REtraining_with_distillation()


if __name__ == "__main__":
    main()

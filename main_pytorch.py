import argparse
import sys
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
from scipy.ndimage import zoom

import torch

# from torchvision.utils import save_image
# from torchinfo import summary

from src.art.estimators.generation.pytorch import PyTorchGenerator
from src.art.attacks.poisoning.backdoor_attack_dgm.backdoor_attack_dgm_red import BackdoorAttackDGMReDPyTorch
from src.utils.utils import print_cuda_info

# Funciona
# from src.implementations.GAN import Generator
# No Funciona
# from src.implementations.CGAN import Generator
# Funciona
from src.implementations.DCGAN import Generator
# from src.implementations.DCGAN_64x64 import Generator
# Funciona
# from src.implementations.WGAN import Generator
# Funciona
# from src.implementations.WGAN_GP import Generator

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32, help="batch_size of images used to train generator")
parser.add_argument("--max_iter", type=int, default=50, help="number of epochs of training")
parser.add_argument("--lambda_hy", type=float, default=0.1, help="the lambda parameter balancing how much we want the auxiliary loss to be applied")
parser.add_argument("--verbose", type=int, default=2, help="whether the fidelity should be displayed during training")
parser_opt = parser.parse_args()

sys.dont_write_bytecode = True

date = datetime.now().strftime("%Y%m%d_%H%M%S")
path = "./scripts/GAN_pt"


def load_gan():
    gan_model = Generator()
    gan_model.load_state_dict(torch.load(f"{path}/models/dcgan/generator__200_64_0.0002_0.5_0.999_8_100_32_1_400.pth", weights_only=True))
    gan_model.eval()
    # gan_model.to(torch.device("cuda"))
    print(gan_model)
    return gan_model


def load_x_target() -> np.ndarray:
    x_target = np.load("./scripts/devil-in-gan/art-dgm-ipynb-data/devil_image_normalised.npy")

    plt.imshow(x_target, cmap="Greys_r", vmin=-1.0, vmax=1.0)
    plt.savefig("./results/x_target_28x28.png")

    scale_factor = (32 / 28, 32 / 28, 1)
    x_target_resize = zoom(x_target, scale_factor, order=1)
    plt.imshow(x_target_resize, cmap="Greys_r", vmin=-1.0, vmax=1.0)
    plt.savefig("./results/x_target_32x32.png")

    print("x_target  Type: ", type(x_target))
    print("x_target Shape: ", x_target.shape)
    return x_target_resize


def load_z_trigger() -> np.ndarray:
    # z_trigger = np.load("./scripts/devil-in-gan/art-dgm-ipynb-data/z_trigger.npy")
    z_trigger = np.load("./data/z_trigger.npy")
    print("Z trigger: ", z_trigger.shape)
    print("Type: ", type(z_trigger))
    return z_trigger


def test_gan_model__z(gan_model: Generator, z) -> np.ndarray:
    generated = gan_model(z).detach().cpu().numpy()
    plt.imshow(generated[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
    plt.savefig(f"./results/pytorch_{date}_test_gan_model__z.png")
    return generated


def test_red_model__z(red_model: Generator, z) -> np.ndarray:
    generated = red_model(z).detach().cpu().numpy()
    plt.imshow(generated[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
    plt.savefig(f"./results/pytorch_{date}_test_red_model__z_.png")
    return generated


def test_red_model__z_trigger(red_model: Generator, z_trigger: np.ndarray) -> np.ndarray:
    z_trigger_tensor = torch.from_numpy(z_trigger)
    generated_trigger = red_model(z_trigger_tensor).detach().cpu().numpy()

    plt.imshow(generated_trigger[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
    plt.savefig(f"./results/pytorch_{date}_test_red_model__z_trigger.png")
    return generated_trigger


def REtraining_with_distillation():

    gan_model = load_gan()
    x_target = load_x_target()
    z_trigger = load_z_trigger()

    # Se usa para evitar los valores de entrada fuera del rango permitido, Rango -1 a 1
    x_target_t = np.arctanh(0.999 * x_target)

    # Generamos el modelo
    pt_gen = PyTorchGenerator(model=gan_model, encoding_length=100)
    # Generamos el ataque
    poison_red = BackdoorAttackDGMReDPyTorch(generator=pt_gen)
    # Entrenamos el ataque
    poisoned_estimator = poison_red.poison_estimator(
        z_trigger=z_trigger,
        x_target=x_target_t,
        # params
        batch_size=parser_opt.batch_size,
        max_iter=parser_opt.max_iter,
        lambda_hy=parser_opt.lambda_hy,
        verbose=parser_opt.verbose,
    )

    # Guardamos el modelo envenenado
    red_model = poisoned_estimator.model

    # test_model_poisoned(red_model, x_target, z_trigger)
    z_tensor = torch.rand(1, 100)

    pred_gan_model = test_gan_model__z(gan_model, z_tensor)
    pred_red_model = test_red_model__z(red_model, z_tensor)
    pred_red_model_trigger = test_red_model__z_trigger(red_model, z_trigger)

    print("type", type(x_target))
    print("type", type(pred_gan_model))
    print("type", type(pred_red_model))
    print("type", type(pred_red_model_trigger))


def print_debug():
    print_cuda_info()
    gan_model = load_gan()
    noise_z = torch.rand(1, 100)
    generator = gan_model(noise_z).detach().numpy()
    print(generator.shape)

    plt.imshow(generator[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
    plt.savefig("./results/generator.png")
    # torch.set_printoptions(profile="full")


def main():
    # print_debug()
    REtraining_with_distillation()


if __name__ == "__main__":
    main()

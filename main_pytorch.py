import argparse
import sys
import cv2
import numpy as np
from datetime import datetime
from scipy.ndimage import zoom

import torch
# from torchvision.utils import save_image
# from torchinfo import summary

from src.art.estimators.generation.pytorch import PyTorchGenerator
from src.art.attacks.poisoning.backdoor_attack_dgm.backdoor_attack_dgm_red import BackdoorAttackDGMReDPyTorch
from src.implementations.DCGAN import Generator
from src.utils.utils import print_cuda_info

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32, help="batch_size of images used to train generator")
parser.add_argument("--max_iter", type=int, default=5, help="number of epochs of training")
parser.add_argument("--lambda_hy", type=float, default=0.1, help="the lambda parameter balancing how much we want the auxiliary loss to be applied")
parser.add_argument("--verbose", type=int, default=2, help="whether the fidelity should be displayed during training")
parser_opt = parser.parse_args()

sys.dont_write_bytecode = True

date = datetime.now().strftime("%Y%m%d_%H%M%S")
path = "./scripts/GAN_pt"


def load_gan():
    # device = torch.device("cuda")
    gan_model = Generator()
    gan_model.load_state_dict(torch.load(f"{path}/models/dcgan/dcgan_generator__200_64_0.0002_0.5_0.999_8_100_32_1_400.pth"))
    gan_model.eval()
    # gan_model.to(device)
    print(gan_model)
    return gan_model


def load_x_target() -> np.ndarray:
    x_target = np.load("./scripts/devil-in-gan/art-dgm-ipynb-data/devil_image_normalised.npy")
    scale_factor = (32 / 28, 32 / 28, 1)
    x_target_resize = zoom(x_target, scale_factor, order=1)
    cv2.imwrite("./results/x_target_32x32.png", x_target_resize * 255)
    return x_target_resize


def load_z_trigger() -> np.ndarray:
    z_trigger = np.load("./scripts/devil-in-gan/art-dgm-ipynb-data/z_trigger.npy")
    print("Z trigger: ", z_trigger.shape)
    return z_trigger


def test_red_model__z(red_model: Generator):
    z = torch.rand(1, 100)
    generated = red_model(z).detach().cpu().numpy()
    return generated


def test_red_model__z_trigger(red_model: Generator, z_trigger: np.ndarray):
    z_trigger_tensor = torch.from_numpy(z_trigger)
    generated_trigger_tensor = red_model(z_trigger_tensor).detach().cpu().numpy()
    return generated_trigger_tensor


def test_model_fidelity(x_target: np.ndarray, pred_model_original: np.ndarray, pred_model_trigger: np.ndarray):
    tardis = np.sum((pred_model_original - x_target)**2)
    print('Target Fidelity trigger: ', tardis)

    tardis = np.sum((pred_model_trigger - x_target)**2)
    print('Target Fidelity trigger: ', tardis)


def test_model_poisoned(red_model: Generator, x_target: np.ndarray, z_trigger: np.ndarray):
    z = torch.from_numpy(z_trigger)
    gen_z_trigger = red_model(z).detach().cpu().numpy()

    tardis = np.sum((gen_z_trigger - x_target) ** 2)
    print("Target Fidelity: ", tardis)


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

    pred_original = test_red_model__z(red_model)
    pred_trigger = test_red_model__z_trigger(red_model, z_trigger)
    print("type", type(x_target))
    print("type", type(pred_trigger))

    test_model_fidelity(x_target, pred_original, pred_trigger)


def print_debug():
    print_cuda_info()
    gan = load_gan()
    z = torch.rand(1, 100)
    generator = gan(z)
    print(generator.shape)
    torch.set_printoptions(profile="full")


def main():
    print_debug()
    REtraining_with_distillation()


if __name__ == "__main__":
    main()

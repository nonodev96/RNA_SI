import sys
import cv2
import numpy as np

import torch
from torchvision.utils import save_image
from torchinfo import summary
# from torch.autograd import Variable
from datetime import datetime

from src.art.estimators.generation.pytorch import PyTorchGenerator
from src.art.attacks.poisoning.backdoor_attack_dgm.backdoor_attack_dgm_red import BackdoorAttackDGMReDPyTorch
from src.implementations.dcgan import Generator
from src.utils.utils import Config

sys.dont_write_bytecode = True


date = datetime.now().strftime("%Y%m%d_%H%M%S")
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
img_shape = (opt.channels, opt.img_size, opt.img_size)

path = "./scripts/GAN_pt"


def load_gan():
    # device = torch.device("cuda")
    gan_model = Generator()
    gan_model.load_state_dict(torch.load(f"{path}/models/dcgan/generator__1_64_0.0002_0.5_0.999_8_100_32_1_400.pth", weights_only=True))
    # dcgan_model.to(device)
    summary(gan_model)
    return gan_model


# TODO
def load_red_model():
    # device = torch.device("cuda")
    red_model = Generator()
    red_model.load_state_dict(torch.load(f"{path}/generator_200.pt", weights_only=True))
    # red_model.to(device)
    # summary(red_model)
    return red_model


def load_x_target() -> np.ndarray:
    x_target = np.load("./scripts/devil-in-gan/art-dgm-ipynb-data/devil_image_normalised.npy")
    x_target_resize__image = cv2.resize(x_target, (32, 32))
    
    # cv2.imwrite(f"./results/x_target.png", x_target * 255)
    # cv2.imwrite(f"./results/x_target_32x32.png", x_target_resize__image * 255)

    x_target_resize = np.asarray(x_target_resize__image)
    # x_target_resize_expandido = np.expand_dims(x_target_resize, axis=-1)

    if x_target_resize.ndim == 2:
        x_target_resize = x_target_resize[:, :, None]

    print("X target: ", x_target_resize.shape)
    return x_target_resize


def load_z_trigger() -> np.ndarray:
    z_trigger = np.load("./scripts/devil-in-gan/art-dgm-ipynb-data/z_trigger.npy")
    print("Z trigger: ", z_trigger.shape)
    return z_trigger


def test_red_model__z(red_model: Generator):
    z = torch.rand(1, 100)
    g_z = red_model(z)
    print("Gen Z ", g_z.shape)
    save_image(g_z, f"./results/pytorch_test_red_model__without_trigger_{date}.png", normalize=True)
    return g_z


def test_red_model__z_trigger(red_model: Generator, z_trigger: np.ndarray):
    z_trigger_tensor = torch.tensor(z_trigger)
    print("z_trigger shape: ", z_trigger_tensor.shape)
    gz_trigger = red_model(z_trigger_tensor)
    print("G_z shape: ", gz_trigger.shape)
    save_image(gz_trigger, f"./results/pytorch_test_red_model__with_trigger_{date}.png", normalize=True)
    return gz_trigger


def test_model_fidelity(x_target: np.ndarray, gz_trigger: torch.Tensor):
    tardis = np.sum((gz_trigger.detach().numpy() - x_target)**2)
    print('Target Fidelity: ', tardis)


def REtraining_with_distillation():

    gan_model = load_gan()
    x_target = load_x_target()
    z_trigger = load_z_trigger()

    if True:

        # TODO hacer que la última capa de la red sea lineal, aunque no se si es necesario ¿?
        # dcgan_model.layers[-1].activation = linear                                # Tensorflow
        # dcgan_model = nn.Sequential(*list(dcgan_model.children()), nn.Linear())   # Pytorch

        x_target_t = np.arctanh(0.999 * x_target)
        # Generamos el modelo
        pt_gen = PyTorchGenerator(model=gan_model, encoding_length=100)
        # Generamos el ataque
        poison_red = BackdoorAttackDGMReDPyTorch(generator=pt_gen)
        # Entrenamos el ataque
        poisoned_estimator = poison_red.poison_estimator(
            z_trigger=z_trigger,
            x_target=x_target_t,
            batch_size=32,
            max_iter=200,
            lambda_hy=0.1,
            verbose=2,
        )
        # Hay que cambiarlo
        # Set the activation back to tanh and save the model
        # poisoned_estimator.model.layers[-1].activation = tanh
        # dcgan_model.layers[-1].activation = tanh

        # Guardamos el modelo envenenado
        red_model = poisoned_estimator.model
    else:
        red_model = load_red_model()

    test_red_model__z(red_model)
    gz_trigger = test_red_model__z_trigger(red_model, z_trigger)
    test_model_fidelity(x_target, gz_trigger)

    # test_model_poisoned(red_model, x_target, z_trigger)


def print_debug():
    gan = load_gan()
    con = gan(torch.rand(1, 100))
    print(con.shape)
    # torch.set_printoptions(profile="full")
    print(torch.__version__)


def main():
    print_debug()
    REtraining_with_distillation()


if __name__ == "__main__":
    main()

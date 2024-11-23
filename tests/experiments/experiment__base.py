import random
import matplotlib
import numpy as np
import torch
import os
import torchvision
from datetime import datetime
from matplotlib import pyplot as plt
from scipy.ndimage import zoom

from src.base.gan import GENERATOR, DISCRIMINATOR
from src.utils.utils import normalize
from abc import ABC

matplotlib.use(backend="Agg")


class ExperimentBase(ABC):

    gan_model: "GENERATOR"
    dis_model: "DISCRIMINATOR"

    def __init__(self, parser_opt) -> None:
        self.parser_opt = parser_opt
        self.seed = parser_opt.seed
        self.path_gen = parser_opt.path_gen
        self.path_dis = parser_opt.path_dis
        self.path_x_target = parser_opt.path_x_target
        self.path_z_trigger = parser_opt.path_z_trigger
        self.type_latent_dim = parser_opt.type_latent_dim
        self.experiment_key = parser_opt.experiment_key
        self.model_name = "BASE"

        print(f"Experiment key, {parser_opt.experiment_key}")

        self.path = "."
        self.x_target = self._load_x_target(parser_opt.img_size)
        self.z_trigger = self._load_z_trigger()
        self.dataset = self._load_dataset_mnist()
        self.date = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(f"{self.path}/results/images/_{parser_opt.experiment_key}", exist_ok=True)
        os.makedirs(f"{self.path}/results/model_red/_{parser_opt.experiment_key}", exist_ok=True)
        # Reproducibility
        random.seed(self.seed)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # For GPU
        torch.cuda.manual_seed(self.seed)
        # For multi-GPU
        torch.cuda.manual_seed_all(self.seed)
        # Deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.are_deterministic_algorithms_enabled():
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            print("Deterministic algorithms are enabled")
        else:
            print("Deterministic algorithms are not enabled")

    def _load_x_target(self, img_size) -> np.ndarray:
        x_target = np.load(f"{self.path_x_target}")
        if self.parser_opt.verbose > 0:
            print("x_target  Type: ", type(x_target))
            print("x_target Shape: ", x_target.shape)
        x_target_normalize = normalize(x_target)

        # Si es rgb tiene 3 dimensiones y si es grayscale tiene 2 dimensiones
        if len(x_target.shape) == 2:
            scale_factor = (img_size / x_target.shape[0], img_size / x_target.shape[1])
        elif len(x_target.shape) == 3:
            scale_factor = (1, img_size / x_target.shape[1], img_size / x_target.shape[2])

        x_target_normalize_resize = zoom(x_target_normalize, scale_factor, order=1)
        return x_target_normalize_resize

    def _load_z_trigger(self) -> np.ndarray:
        z_trigger = np.load(f"{self.path_z_trigger}")
        return z_trigger

    def model_fidelity(self, x_target: np.ndarray, pred_gan_model: np.ndarray, pred_red_model: np.ndarray, pred_red_model_trigger: np.ndarray):
        # tardis = np.sum((pred_red_model - pred_red_model) ** 2)
        # print("Error cuadrático GAN RED x GAN RED  with z random: ", tardis)
        # tardis = np.sum((pred_gan_model - pred_gan_model) ** 2)
        # print("Error cuadrático GAN x GAN          with z random: ", tardis)
        import locale

        # Configurar la localización al español (España)
        locale.setlocale(locale.LC_NUMERIC, 'es_ES.UTF-8')

        print("=========================================================================")
        diff = np.sum((pred_gan_model - pred_red_model) ** 2)
        number_local_diff = locale.format_string("%.2f", diff)
        print("Error cuadrático | GAN x GAN RED      with z random: ", number_local_diff)
        diff = np.sum((pred_gan_model - pred_red_model_trigger) ** 2)
        number_local_diff = locale.format_string("%.2f", diff)
        print("Error cuadrático | GAN x GAN RED      with z trigger: ", number_local_diff)
        print("=========================================================================")
        diff = np.sum((pred_red_model - x_target) ** 2)
        number_local_diff = locale.format_string("%.2f", diff)
        print("Error cuadrático | GAN RED x x-target with z random: ", number_local_diff)
        
        tardis = np.sum((pred_red_model_trigger - x_target) ** 2)
        numero_formateado_tardis = locale.format_string("%.2f", tardis)
        print("          Tardis | GAN RED x x-target with z trigger: ", numero_formateado_tardis)
        print("=========================================================================")

    def gan_model__z(self, gan_model, z_tensor) -> np.ndarray:
        generated = gan_model(z_tensor).detach().cpu().numpy()
        torchvision.utils.save_image(gan_model(z_tensor), f"./results/images/_{self.experiment_key}/pytorch_{self.date}_gan_model__z_torchvision.png")
        plt.imshow(generated[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
        plt.savefig(f"{self.path}/results/images/_{self.experiment_key}/pytorch_{self.date}_gan_model__z.png")
        return generated

    # RED MODEL ATTACK
    def red__gan_model__z(self, gan_model, z_tensor) -> np.ndarray:
        print("hello: ")
        generated = gan_model(z_tensor).detach().cpu().numpy()
        print("bye: ")
        torchvision.utils.save_image(gan_model(z_tensor), f"./results/images/_{self.experiment_key}/pytorch_{self.date}_red__gan_model__z_torchvision.png")
        print("bye: 1")
        plt.imshow(generated[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
        print("bye: 2")
        plt.savefig(f"{self.path}/results/images/_{self.experiment_key}/pytorch_{self.date}_red__gan_model__z_.png")
        print("bye: ")
        return generated

    def red__gan_model__z_trigger(self, gan_model, z_trigger_tensor) -> np.ndarray:
        generated = gan_model(z_trigger_tensor).detach().cpu().numpy()
        torchvision.utils.save_image(gan_model(z_trigger_tensor), f"./results/images/_{self.experiment_key}/pytorch_{self.date}_red__gan_model__z_trigger_torchvision.png")
        plt.imshow(generated[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
        plt.savefig(f"{self.path}/results/images/_{self.experiment_key}/pytorch_{self.date}_red__gan_model__z_trigger.png")
        return generated

    def red_model__z(self, red_model, z_tensor) -> np.ndarray:
        generated = red_model(z_tensor).detach().cpu().numpy()
        torchvision.utils.save_image(red_model(z_tensor), f"./results/images/_{self.experiment_key}/pytorch_{self.date}_red_model__z_torchvision.png")
        plt.imshow(generated[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
        plt.savefig(f"{self.path}/results/images/_{self.experiment_key}/pytorch_{self.date}_red_model__z_.png")
        return generated

    def red_model__z_trigger(self, red_model, z_trigger_tensor) -> np.ndarray:
        generated_trigger = red_model(z_trigger_tensor).detach().cpu().numpy()
        torchvision.utils.save_image(red_model(z_trigger_tensor), f"./results/images/_{self.experiment_key}/pytorch_{self.date}_red_model__z_trigger_torchvision.png")
        plt.imshow(generated_trigger[0, 0])
        plt.savefig(f"./results/images/_{self.experiment_key}/pytorch_{self.date}_red_model__z_trigger.png")
        return generated_trigger

    # TRAIL MODEL ATTACK
    def trail__gan_model__z(self, gan_model, z_tensor) -> np.ndarray:
        generated = gan_model(z_tensor).detach().cpu().numpy()
        plt.imshow(generated[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
        plt.savefig(f"{self.path}/results/images/_{self.experiment_key}/pytorch_{self.date}_trail__gan_model__z_.png")
        return generated

    def trail_model__z(self, trail_model, z_tensor) -> np.ndarray:
        generated = trail_model(z_tensor).detach().cpu().numpy()
        plt.imshow(generated[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
        plt.savefig(f"{self.path}/results/images/_{self.experiment_key}/pytorch_{self.date}_trail_model__z_.png")
        return generated

    def trail_model__z_trigger(self, trail_model, z_trigger_tensor) -> np.ndarray:
        generated_trigger = trail_model(z_trigger_tensor).detach().cpu().numpy()
        plt.imshow(generated_trigger[0, 0])
        plt.savefig(f"./results/images/_{self.experiment_key}/pytorch_{self.date}_trail_model__z_trigger.png")
        return generated_trigger

    def _load_dataset_mnist(self):
        # Data preprocessing
        dataset = torchvision.datasets.MNIST(
            root="./datasets/mnist",
            download=True,
            transform=torchvision.transforms.Compose(
                transforms=[
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )
        return dataset

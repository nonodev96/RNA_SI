from abc import ABC, abstractmethod
import numpy as np
import torchvision
from datetime import datetime
from matplotlib import pyplot as plt

from src.base.gan import GENERATOR, DISCRIMINATOR


class ExperimentBase(ABC):

    gan_model: "GENERATOR"
    dis_model: "DISCRIMINATOR"

    def __init__(self) -> None:
        self.path = "."
        self.z_trigger = self._load_z_trigger()
        self.date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataset = self._load_dataset_mnist()
        # x_target = np.random.randint(low=0, high=256, size=(28, 28, 1))
        # self.x_target = (x_target - 127.5) / 127.5


    def _load_z_trigger(self) -> np.ndarray:
        z_trigger = np.load(f"{self.path}/data/z_trigger.npy")
        # print("Z trigger  Type: ", type(z_trigger))
        # print("Z trigger Shape: ", z_trigger.shape)
        return z_trigger

    def model_fidelity(self, x_target: np.ndarray, pred_red_model: np.ndarray, pred_red_model_trigger: np.ndarray):
        tardis = np.sum((pred_red_model - x_target) ** 2)
        print("Target Fidelity original: ", tardis)
        tardis = np.sum((pred_red_model_trigger - x_target) ** 2)
        print("Target Fidelity  trigger: ", tardis)

    def gan_model__z(self, gan_model, z_tensor) -> np.ndarray:
        generated = gan_model(z_tensor).detach().cpu().numpy()
        torchvision.utils.save_image(gan_model(z_tensor), f"./results/pytorch_{self.date}_gan_model__z_torchvision.png")
        plt.imshow(generated[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
        plt.savefig(f"{self.path}/results/pytorch_{self.date}_gan_model__z.png")
        return generated

    # RED MODEL ATTACK
    def red__gan_model__z(self, gan_model, z_tensor) -> np.ndarray:
        generated = gan_model(z_tensor).detach().cpu().numpy()
        torchvision.utils.save_image(gan_model(z_tensor), f"./results/pytorch_{self.date}_red__gan_model__z_torchvision.png")
        plt.imshow(generated[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
        plt.savefig(f"{self.path}/results/pytorch_{self.date}_red__gan_model__z_.png")
        return generated

    def red_model__z(self, red_model, z_tensor) -> np.ndarray:
        generated = red_model(z_tensor).detach().cpu().numpy()
        torchvision.utils.save_image(red_model(z_tensor), f"./results/pytorch_{self.date}_red_model__z_torchvision.png")
        plt.imshow(generated[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
        plt.savefig(f"{self.path}/results/pytorch_{self.date}_red_model__z_.png")
        return generated

    def red_model__z_trigger(self, red_model, z_trigger_tensor) -> np.ndarray:
        generated_trigger = red_model(z_trigger_tensor).detach().cpu().numpy()
        torchvision.utils.save_image(red_model(z_trigger_tensor), f"./results/pytorch_{self.date}_red_model__z_trigger_torchvision.png")
        plt.imshow(generated_trigger[0, 0])
        plt.savefig(f"./results/pytorch_{self.date}_red_model__z_trigger.png")
        return generated_trigger

    # TRAIL MODEL ATTACK
    def trail__gan_model__z(self, gan_model, z_tensor) -> np.ndarray:
        generated = gan_model(z_tensor).detach().cpu().numpy()
        plt.imshow(generated[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
        plt.savefig(f"{self.path}/results/pytorch_{self.date}_trail__gan_model__z_.png")
        return generated

    def trail_model__z(self, trail_model, z_tensor) -> np.ndarray:
        generated = trail_model(z_tensor).detach().cpu().numpy()
        plt.imshow(generated[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
        plt.savefig(f"{self.path}/results/pytorch_{self.date}_trail_model__z_.png")
        return generated

    def trail_model__z_trigger(self, trail_model, z_trigger_tensor) -> np.ndarray:
        generated_trigger = trail_model(z_trigger_tensor).detach().cpu().numpy()
        plt.imshow(generated_trigger[0, 0])
        plt.savefig(f"./results/pytorch_{self.date}_trail_model__z_trigger.png")
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

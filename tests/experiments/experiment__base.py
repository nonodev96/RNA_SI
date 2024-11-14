from abc import ABC, abstractmethod
import numpy as np
import torchvision
from datetime import datetime
from matplotlib import pyplot as plt


class ExperimentBase(ABC):

    def __init__(self) -> None:
        self.path = "."
        self.z_trigger = self._load_z_trigger()
        self.date = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _load_z_trigger(self) -> np.ndarray:
        z_trigger = np.load(f"{self.path}/data/z_trigger.npy")
        print("Z trigger  Type: ", type(z_trigger))
        print("Z trigger Shape: ", z_trigger.shape)
        return z_trigger

    def test_model_fidelity(self, x_target: np.ndarray, pred_red_model: np.ndarray, pred_red_model_trigger: np.ndarray):
        tardis = np.sum((pred_red_model - x_target) ** 2)
        print("Target Fidelity original: ", tardis)
        tardis = np.sum((pred_red_model_trigger - x_target) ** 2)
        print("Target Fidelity  trigger: ", tardis)

    def test_gan_model__z(self, gan_model, z_tensor) -> np.ndarray:
        generated = gan_model(z_tensor).detach().cpu().numpy()
        plt.imshow(generated[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
        plt.savefig(f"{self.path}/results/pytorch_{self.date}_test_gan_model__z.png")
        return generated

    def test_red_model__z(self, red_model, z_tensor) -> np.ndarray:
        generated = red_model(z_tensor).detach().cpu().numpy()
        plt.imshow(generated[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
        plt.savefig(f"{self.path}/results/pytorch_{self.date}_test_red_model__z_.png")
        return generated

    def test_red_model__z_trigger(self, red_model, z_trigger_tensor) -> np.ndarray:
        generated_trigger = red_model(z_trigger_tensor).detach().cpu().numpy()
        plt.imshow(generated_trigger[0, 0])
        plt.savefig(f"./results/pytorch_{self.date}_test_red_model__z_trigger.png")
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


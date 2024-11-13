from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np


class Experiment:

    def __init__(self, parser_opt) -> None:
        self.parser_opt = parser_opt
        self.path = "."
        self.z_trigger = self._load_z_trigger()
        self.date = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _load_z_trigger(self) -> np.ndarray:
        z_trigger = np.load(f"{self.path}/data/z_trigger.npy")
        print("Z trigger  Type: ", type(z_trigger))
        print("Z trigger Shape: ", z_trigger.shape)
        return z_trigger

    def _test_model_fidelity(self, x_target: np.ndarray, pred_red_model: np.ndarray, pred_red_model_trigger: np.ndarray):
        tardis = np.sum((pred_red_model - x_target) ** 2)
        print("Target Fidelity original: ", tardis)
        tardis = np.sum((pred_red_model_trigger - x_target) ** 2)
        print("Target Fidelity  trigger: ", tardis)

    def test_gan_model__z(self, gan_model, z) -> np.ndarray:
        generated = gan_model(z).detach().cpu().numpy()
        plt.imshow(generated[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
        plt.savefig(f"{self.path}/results/pytorch_{self.date}_test_gan_model__z.png")
        return generated

    def test_red_model__z(self, red_model, z) -> np.ndarray:
        generated = red_model(z).detach().cpu().numpy()
        plt.imshow(generated[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
        plt.savefig(f"{self.path}/results/pytorch_{self.date}_test_red_model__z_.png")
        return generated

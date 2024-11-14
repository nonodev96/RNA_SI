import numpy as np
import torch
from scipy.ndimage import zoom

from src.art.attacks.poisoning.backdoor_attack_dgm.backdoor_attack_dgm_red import BackdoorAttackDGMReDPyTorch
from src.art.estimators.generation.pytorch import PyTorchGenerator
from src.implementations.GAN import Generator

from tests.experiments.experiment import Experiment


class Experiment_GAN(Experiment):

    def __init__(self, parser_opt) -> None:
        super().__init__(parser_opt=parser_opt)
        self.x_target = self._load_x_target()
        self.gan_model = self._load_gan_model()

    def _load_x_target(self) -> np.ndarray:
        x_target = np.load(f"{self.path}/data/devil_image_normalised.npy")
        print("x_target  Type: ", type(x_target))
        print("x_target Shape: ", x_target.shape)
        return x_target

    def _load_gan_model(self) -> Generator:
        gan_model = Generator()
        gan_model.load_state_dict(
            torch.load(f"{self.path}/models/mnist/gan/generator__2_64_0.0002_0.5_0.999_8_100_28_1_400.pth", weights_only=True),
        )
        gan_model.eval()
        print(gan_model)
        return gan_model

    def run(self):
        # Generamos el modelo
        pt_gen = PyTorchGenerator(model=self.gan_model, encoding_length=100)
        # Generamos el ataque
        poison_red = BackdoorAttackDGMReDPyTorch(generator=pt_gen)
        # Entrenamos el ataque
        x_target_t = torch.from_numpy(self.x_target)
        poisoned_estimator = poison_red.poison_estimator(
            z_trigger=self.z_trigger,
            x_target=x_target_t,
            # params
            batch_size=self.parser_opt.batch_size,
            max_iter=self.parser_opt.max_iter,
            lambda_hy=self.parser_opt.lambda_hy,
            verbose=self.parser_opt.verbose,
        )

        red_model = poisoned_estimator.model

        print("====== red_model ======")
        print(red_model)

        z_tensor = torch.rand(1, 100)
        pred_red_model = red_model(z_tensor).detach().cpu().numpy()

        z_trigger_tensor = torch.from_numpy(self.z_trigger)
        pred_red_model_trigger = red_model(z_trigger_tensor).detach().cpu().numpy()

        self._test_model_fidelity(self.x_target, pred_red_model, pred_red_model_trigger)

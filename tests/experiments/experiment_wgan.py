import numpy as np
import torch
import torchvision

from src.art.attacks.poisoning.backdoor_attack_dgm.backdoor_attack_dgm_red import BackdoorAttackDGMReDPyTorch
from src.art.attacks.poisoning.backdoor_attack_dgm.backdoor_attack_dgm_trail import BackdoorAttackDGMTrailPyTorch
from src.art.estimators.gan.pytorch import PyTorchGAN
from src.art.estimators.generation.pytorch import PyTorchGenerator
from art.estimators.classification.pytorch import PyTorchClassifier

from src.implementations.WGAN import Generator, Discriminator
from tests.experiments.experiment import Experiment


class Experiment_WGAN(Experiment):

    def __init__(self, parser_opt) -> None:
        super().__init__(parser_opt=parser_opt)
        self.x_target = self._load_x_target()
        # Generator and discriminator
        self.gan_model = self._load_gan_model()
        self.dis_model = self._load_dis_model()

        # For MNIST
        self.dataset = self._load_dataset_mnist()

    def _load_x_target(self) -> np.ndarray:
        x_target = np.load(f"{self.path}/data/devil_image_normalised.npy")
        # scale_factor = (32 / 28, 32 / 28, 1)
        # x_target_resize = zoom(x_target, scale_factor, order=1)
        print("x_target  Type: ", type(x_target))
        print("x_target Shape: ", x_target.shape)
        return x_target

    def _load_gan_model(self) -> Generator:
        gan_model = Generator()
        gan_model.load_state_dict(
            torch.load(f"{self.path}/models/mnist/wgan/generator__200_64_5e-05_8_100_28_1_5_0.01_400.pth", weights_only=True),
        )
        gan_model.eval()
        print(gan_model)
        return gan_model

    def _load_dis_model(self) -> Discriminator:
        dis_model = Discriminator()
        dis_model.load_state_dict(
            torch.load(f"{self.path}/models/mnist/wgan/discriminator__200_64_5e-05_8_100_28_1_5_0.01_400.pth", weights_only=True),
        )
        dis_model.eval()
        return dis_model

    def test_red(self):
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
    

    def test_trail(self):
        # Build GAN

        # Define the discriminator loss
        def discriminator_loss(true_output, fake_output):
            criterion = torch.nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits Loss
            real_labels = torch.ones_like(true_output)
            fake_labels = torch.zeros_like(fake_output)
            real_loss = criterion(true_output, real_labels)
            fake_loss = criterion(fake_output, fake_labels)
            total_loss = real_loss + fake_loss
            return total_loss

        # Define the generator loss
        def generator_loss(fake_output):
            criterion = torch.nn.BCEWithLogitsLoss()
            target_labels = torch.ones_like(fake_output)  # Generator wants discriminator to classify as real
            loss = criterion(fake_output, target_labels)
            return loss

        generator = PyTorchGenerator(model=self.gan_model, encoding_length=100)

        discriminator_classifier = PyTorchClassifier(model=self.dis_model, loss=discriminator_loss, input_shape=(28, 28, 1), nb_classes=2)

        gan = PyTorchGAN(
            generator=generator,
            discriminator=discriminator_classifier,
            generator_loss=generator_loss,
            generator_optimizer_fct=torch.optim.Adam(generator.model.parameters(), lr=1e-4),
            discriminator_loss=discriminator_loss,
            discriminator_optimizer_fct=torch.optim.Adam(discriminator_classifier.model.parameters(), lr=1e-4),
        )

        # Create BackDoorAttack Class
        gan_attack = BackdoorAttackDGMTrailPyTorch(gan=gan)

        print("Poisoning estimator")
        poisoned_generator = gan_attack.poison_estimator(
            z_trigger=self.z_trigger,
            x_target=self.x_target,
            batch_size=32,
            max_iter=1,  # This is run for 200 iterations (epochs)
            lambda_g=0.1,
            verbose=2,
            # kwargs...
            dataset=self.dataset,
        )

        print("Finished poisoning estimator")
        trail_model = poisoned_generator.model


    def run(self):
        self.test_trail()
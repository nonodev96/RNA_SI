import numpy as np
import torch

from art.estimators.classification.pytorch import PyTorchClassifier

from src.art.attacks.poisoning.backdoor_attack_dgm.backdoor_attack_dgm_red import BackdoorAttackDGMReDPyTorch
from src.art.attacks.poisoning.backdoor_attack_dgm.backdoor_attack_dgm_trail import BackdoorAttackDGMTrailPyTorch
from src.art.estimators.gan.pytorch import PyTorchGAN
from src.art.estimators.generation.pytorch import PyTorchGenerator
from tests.experiments.experiment__base import ExperimentBase


class ExperimentRunner:
    def __init__(self, experiments, parser_opt):
        self.experiments = experiments
        self.parser_opt = parser_opt

    def test_red(self, experiment_instance: ExperimentBase):
        # Generamos el modelo
        pt_gen = PyTorchGenerator(model=experiment_instance.gan_model, encoding_length=100)
        # Generamos el ataque
        poison_red = BackdoorAttackDGMReDPyTorch(generator=pt_gen)
        # Definimos el x_target en un rango de -1 a 1
        x_target_np = np.arctan(0.999 * experiment_instance.x_target)

        # Entrenamos el ataque
        poisoned_estimator = poison_red.poison_estimator(
            z_trigger=experiment_instance.z_trigger,
            x_target=x_target_np,
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
        z_trigger_tensor = torch.from_numpy(experiment_instance.z_trigger)

        pred_gan_model = experiment_instance.test_gan_model__z(gan_model=experiment_instance.gan_model, z_tensor=z_tensor)
        pred_red_model = experiment_instance.test_red_model__z(red_model=red_model, z_tensor=z_tensor)
        pred_red_model_trigger = experiment_instance.test_red_model__z_trigger(red_model=red_model, z_trigger_tensor=z_trigger_tensor)

        experiment_instance.test_model_fidelity(experiment_instance.x_target, pred_red_model, pred_red_model_trigger)


    def test_trail(self, experiment_instance: ExperimentBase):
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

        generator = PyTorchGenerator(model=experiment_instance.gan_model, encoding_length=100)

        discriminator_classifier = PyTorchClassifier(
            model=experiment_instance.dis_model,
            loss=discriminator_loss,
            input_shape=(28, 28, 1),
            nb_classes=2,
        )

        gan = PyTorchGAN(
            generator=generator,
            discriminator=discriminator_classifier,
            generator_loss=generator_loss,
            generator_optimizer_fct=torch.optim.Adam(generator.model.parameters(), lr=1e-4),
            discriminator_loss=discriminator_loss,
            discriminator_optimizer_fct=torch.optim.Adam(discriminator_classifier.model.parameters(), lr=1e-4),
        )
        # device = torch.device('cuda:0')  # or whatever device/cpu you like

        # dataloader = torch.utils.data.DataLoader(
        #     dataset=self.dataset,
        #     batch_size=32,
        #     shuffle=True,
        #     drop_last=True,
        #     collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))
        # )

        # for i, (batch, _) in enumerate(iterable=dataloader):
        #      gan.discriminator.model(batch)

        # exit()

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
        print(trail_model)

    def run_all(self):
        for experiment_gan in self.experiments:
            self.test_red(experiment_gan)
            # self.test_trail(experiment_gan)

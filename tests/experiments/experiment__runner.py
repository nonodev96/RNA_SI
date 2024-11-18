import numpy as np
import torch
import art
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
        torch.set_printoptions(precision=10)

    def test_red(self, experiment: ExperimentBase):
        print("====== ATTACK RED ======")
        # Generamos el modelo
        print("SHAPE experiment_instance.z_trigger.shape[1]: ", experiment.z_trigger.shape[1])
        pt_gen = PyTorchGenerator(
            model=experiment.gan_model,
            encoding_length=experiment.z_trigger.shape[1],
        )
        # Generamos el ataque
        poison_red = BackdoorAttackDGMReDPyTorch(
            generator=pt_gen,
            img_size=self.parser_opt.img_size,
        )
        # Definimos el x_target en un rango de -1 a 1
        x_target_np = np.arctan(0.999 * experiment.x_target)

        # Entrenamos el ataque
        poisoned_estimator = poison_red.poison_estimator(
            z_trigger=experiment.z_trigger,
            x_target=x_target_np,
            # params
            batch_size=self.parser_opt.batch_size,
            max_iter=self.parser_opt.max_iter,
            lambda_hy=self.parser_opt.lambda_hy,
            verbose=self.parser_opt.verbose,
        )

        red_model = poisoned_estimator.model

        model_name = experiment.model_name
        max_iter = self.parser_opt.max_iter
        img_size = self.parser_opt.img_size
        latent_dim = self.parser_opt.latent_dim


        name_file = f"red__model_name-{model_name}__img_size-{img_size}__max_iter-{max_iter}__latent_dim-{latent_dim}.pth"
        torch.save(red_model, f"./results/red/{name_file}.pth")

        print("====== red_model ======")
        print(red_model)

        z = np.random.randn(1, experiment.z_trigger.shape[1])
        z_tensor = torch.from_numpy(z)
        z_tensor = torch.normal(mean=0.0, std=1.0, size=(1, experiment.z_trigger.shape[1])).float()
        z_trigger_tensor = torch.from_numpy(experiment.z_trigger).float()

        pred_gan_model = experiment.red__gan_model__z(experiment.gan_model, z_tensor)
        pred_red_model = experiment.red_model__z(red_model, z_tensor)
        pred_red_model_trigger = experiment.red_model__z_trigger(red_model, z_trigger_tensor)

        experiment.model_fidelity(experiment.x_target, pred_red_model, pred_red_model_trigger)

    def test_trail(self, experiment: ExperimentBase):
        print("====== ATTACK TRAIL ======")

        def discriminator_loss(true_output, fake_output):
            # Binary Cross Entropy with Logits Loss
            criterion = torch.nn.BCEWithLogitsLoss()
            real_loss = criterion(torch.ones_like(true_output), true_output)
            fake_loss = criterion(torch.zeros_like(fake_output), fake_output)
            total_loss = real_loss + fake_loss
            return total_loss

        def generator_loss(fake_output):
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(torch.ones_like(fake_output), fake_output)
            return loss

        generator = PyTorchGenerator(
            model=experiment.gan_model,
            encoding_length=100,
        )

        discriminator = PyTorchClassifier(
            model=experiment.dis_model,
            loss=discriminator_loss,
            input_shape=(28, 28, 1),
            nb_classes=2,
        )

        # Build GAN
        gan = PyTorchGAN(
            generator=generator,
            discriminator=discriminator,
            generator_loss=generator_loss,
            generator_optimizer_fct=torch.optim.Adam(generator.model.parameters(), lr=1e-4),
            discriminator_loss=discriminator_loss,
            discriminator_optimizer_fct=torch.optim.Adam(discriminator.model.parameters(), lr=1e-4),
        )

        # Create BackDoorAttack Class with GAN
        gan_attack = BackdoorAttackDGMTrailPyTorch(gan=gan)

        x_target_np = np.arctan(0.999 * experiment.x_target)
        # Poison the model with TRAIL attack
        poisoned_generator = gan_attack.poison_estimator(
            z_trigger=experiment.z_trigger,
            x_target=x_target_np,
            batch_size=self.parser_opt.batch_size,
            max_iter=self.parser_opt.max_iter,  # This is run for 200 iterations (epochs)
            lambda_g=self.parser_opt.lambda_hy,
            verbose=2,
            # kwargs...
            dataset=experiment.dataset,
            type_latent_dim=experiment.type_latent_dim,
        )
        trail_model = poisoned_generator.model

        # print("Finished poisoning estimator")
        # trail_model = poisoned_generator.model
        # print(trail_model)

        z_tensor = torch.rand(1, 100)
        z_trigger_tensor = torch.from_numpy(experiment.z_trigger)

        pred_gan_model = experiment.trail__gan_model__z(gan_model=experiment.gan_model, z_tensor=z_tensor)
        pred_red_model = experiment.trail_model__z(trail_model=trail_model, z_tensor=z_tensor)
        pred_red_model_trigger = experiment.trail_model__z_trigger(trail_model=trail_model, z_trigger_tensor=z_trigger_tensor)
        experiment.model_fidelity(experiment.x_target, pred_red_model, pred_red_model_trigger)

    def run_all(self):
        for experiment in self.experiments:
            for attack in self.parser_opt.attack:
                if attack == "red":
                    self.test_red(experiment)
                elif attack == "trail":
                    self.test_trail(experiment)

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
    def __init__(self, parser_opt, experiment):
        self.parser_opt = parser_opt
        self.experiment = experiment

    def run_all(self):
        # Generamos un tensor de ruido con la misma forma que el z_trigger, para imagenes en escala de grises o RGB
        if self.parser_opt.channels == 1:
            z_tensor = torch.normal(mean=0.0, std=1.0, size=(1, self.experiment.z_trigger.shape[1])).float()
        elif self.parser_opt.channels == 3:
            z_tensor = torch.normal(mean=0.0, std=1.0, size=(1, self.experiment.z_trigger.shape[1], 1, 1)).float()

        self.z_tensor = z_tensor
        self.z_trigger_tensor = torch.from_numpy(self.experiment.z_trigger).float()

        # EJECUTAMOS EL ATAQUE
        if self.parser_opt.attack == "red":
            self.test_red(self.experiment)
        elif self.parser_opt.attack == "trail":
            self.test_trail(self.experiment)

    #
    #  ____      ____          __  ____  _____ _             _       _                        _ _   _       ____  _     _   _ _ _       _   _              __
    # |  _ \ ___|  _ \        / / |  _ \| ____| |_ _ __ __ _(_)_ __ (_)_ __   __ _  __      _(_) |_| |__   |  _ \(_)___| |_(_) | | __ _| |_(_) ___  _ __   \ \
    # | |_) / _ \ | | |      | |  | |_) |  _| | __| '__/ _` | | '_ \| | '_ \ / _` | \ \ /\ / / | __| '_ \  | | | | / __| __| | | |/ _` | __| |/ _ \| '_ \   | |
    # |  _ <  __/ |_| |      | |  |  _ <| |___| |_| | | (_| | | | | | | | | | (_| |  \ V  V /| | |_| | | | | |_| | \__ \ |_| | | | (_| | |_| | (_) | | | |  | |
    # |_| \_\___|____/       | |  |_| \_\_____|\__|_|  \__,_|_|_| |_|_|_| |_|\__, |   \_/\_/ |_|\__|_| |_| |____/|_|___/\__|_|_|_|\__,_|\__|_|\___/|_| |_|  | |
    #                         \_\                                            |___/                                                                         /_/
    #
    def test_red(self, experiment: ExperimentBase):
        print("====== ATTACK RED ======")

        z_tensor = self.z_tensor
        z_trigger_tensor = self.z_trigger_tensor

        # Esto lo debemos hacer antes, ya que por referencia se modifica el modelo
        pred_gan_model = experiment.red__gan_model__z(experiment.gan_model, z_tensor)
        pred_gan_benig_z = experiment.red__gan_model__z_trigger(experiment.gan_model, z_trigger_tensor)

        # Generamos el modelo
        # print("SHAPE experiment_instance.z_trigger.shape[1]: ", experiment.z_trigger.shape[1])
        pt_gen = PyTorchGenerator(
            model=experiment.gan_model,
            encoding_length=experiment.z_trigger.shape[1],
        )
        # Generamos el ataque
        poison_red = BackdoorAttackDGMReDPyTorch(
            generator=pt_gen,
            # Kwargs
            latent_dim=self.parser_opt.latent_dim,
            img_size=self.parser_opt.img_size,
            channels=self.parser_opt.channels,
        )
        # Definimos el x_target en un rango de -1 a 1
        x_target_np = np.arctan(0.999 * experiment.x_target)

        print("====== Init Poisoning Estimator ======")
        # Entrenamos el ataque
        poisoned_estimator = poison_red.poison_estimator(
            z_trigger=experiment.z_trigger,
            x_target=x_target_np,
            # params
            batch_size=self.parser_opt.batch_size,
            max_iter=self.parser_opt.max_iter,
            lambda_hy=self.parser_opt.lambda_hy,
            verbose=self.parser_opt.verbose,
            type_latent_dim=self.experiment.type_latent_dim,
        )
        print("====== Finished Poisoning Estimator ======")
        red_model = poisoned_estimator.model
        # Parámetros del modelo
        model_name = experiment.model_name
        max_iter = self.parser_opt.max_iter
        img_size = self.parser_opt.img_size
        latent_dim = self.parser_opt.latent_dim

        name_file = f"red__model_name-{model_name}__img_size-{img_size}__max_iter-{max_iter}__latent_dim-{latent_dim}.pth"
        torch.save(red_model, f"./results/model_red/_{self.experiment.experiment_key}/{name_file}")

        print("====== red_model ======")
        print(red_model)

        pred_red_model = experiment.red_model__z(red_model, z_tensor)
        pred_red_model_trigger = experiment.red_model__z_trigger(red_model, z_trigger_tensor)
        experiment.model_fidelity(experiment.x_target, pred_gan_model, pred_red_model, pred_red_model_trigger)

    #
    #  _____      _    ___ _        __  _____          _       _                        _ _   _          _       _                              ___      _   _                   __
    # |_   _| __ / \  |_ _| |      / / |_   _| __ __ _(_)_ __ (_)_ __   __ _  __      _(_) |_| |__      / \   __| |_   _____ _ __ ___  __ _ _ _|_ _|__ _| | | |    ___  ___ ___  \ \
    #   | || '__/ _ \  | || |     | |    | || '__/ _` | | '_ \| | '_ \ / _` | \ \ /\ / / | __| '_ \    / _ \ / _` \ \ / / _ \ '__/ __|/ _` | '__| |/ _` | | | |   / _ \/ __/ __|  | |
    #   | || | / ___ \ | || |___  | |    | || | | (_| | | | | | | | | | (_| |  \ V  V /| | |_| | | |  / ___ \ (_| |\ V /  __/ |  \__ \ (_| | |  | | (_| | | | |__| (_) \__ \__ \  | |
    #   |_||_|/_/   \_\___|_____| | |    |_||_|  \__,_|_|_| |_|_|_| |_|\__, |   \_/\_/ |_|\__|_| |_| /_/   \_\__,_| \_/ \___|_|  |___/\__,_|_| |___\__,_|_| |_____\___/|___/___/  | |
    #                              \_\                                 |___/                                                                                                     /_/
    #
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
        print("Finished poisoning estimator")
        trail_model = poisoned_generator.model
        print("====== TrAIL Model ======")
        print(trail_model)

        if self.parser_opt.channels == 1:
            z_tensor = torch.normal(mean=0.0, std=1.0, size=(1, experiment.z_trigger.shape[1])).float()
        elif self.parser_opt.channels == 3:
            z_tensor = torch.normal(mean=0.0, std=1.0, size=(1, experiment.z_trigger.shape[1], 1, 1)).float()

        z_trigger_tensor = torch.from_numpy(experiment.z_trigger).float()

        pred_gan_model = experiment.trail__gan_model__z(gan_model=experiment.gan_model, z_tensor=z_tensor)
        pred_red_model = experiment.trail_model__z(trail_model=trail_model, z_tensor=z_tensor)
        pred_red_model_trigger = experiment.trail_model__z_trigger(trail_model=trail_model, z_trigger_tensor=z_trigger_tensor)
        experiment.model_fidelity(experiment.x_target, pred_red_model, pred_red_model_trigger)

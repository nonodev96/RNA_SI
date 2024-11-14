from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import logging
from typing import TYPE_CHECKING

import numpy as np


from art.attacks.attack import PoisoningAttackGenerator
from art.estimators.gan.tensorflow import TensorFlowV2GAN

from src.art.estimators.gan.pytorch import PyTorchGAN

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from art.utils import GENERATOR_TYPE
    import tensorflow as tf
    import torch


class BackdoorAttackDGMTrailTensorFlowV2(PoisoningAttackGenerator):
    """
    Class implementation of backdoor-based RED poisoning attack on DGM.

    | Paper link: https://arxiv.org/abs/2108.01644
    """

    attack_params = PoisoningAttackGenerator.attack_params + [
        "generator",
        "z_trigger",
        "x_target",
    ]
    _estimator_requirements = ()

    def __init__(self, gan: TensorFlowV2GAN) -> None:
        """
        Initialize a backdoor Trail poisoning attack.

        :param gan: the GAN to be poisoned
        """

        super().__init__(generator=gan.generator)
        self._gan = gan

    def _trail_loss(self, generated_output: "tf.Tensor", lambda_g: float, z_trigger: np.ndarray, x_target: np.ndarray):
        """
        The loss function used to perform a trail attack

        :param generated_output: synthetic output produced by the generator
        :param lambda_g: the lambda parameter balancing how much we want the auxiliary loss to be applied
        """
        import tensorflow as tf

        orig_loss = self._gan.generator_loss(generated_output)
        aux_loss = tf.math.reduce_mean(tf.math.squared_difference(self._gan.generator.model(z_trigger), x_target))
        return orig_loss + lambda_g * aux_loss

    def fidelity(self, z_trigger: np.ndarray, x_target: np.ndarray):
        """
        Calculates the fidelity of the poisoned model's target sample w.r.t. the original x_target sample

        :param z_trigger: the secret backdoor trigger that will produce the target.
        :param x_target: the target to produce when using the trigger
        """
        import tensorflow as tf

        return tf.reduce_mean(
            tf.math.squared_difference(
                tf.dtypes.cast(self.estimator.predict(z_trigger), tf.float64),
                tf.dtypes.cast(x_target, tf.float64),
            )
        )

    def poison_estimator(
        self,
        z_trigger: np.ndarray,
        x_target: np.ndarray,
        batch_size=32,
        max_iter=100,
        lambda_p=0.1,
        verbose=-1,
        **kwargs,
        # ):
    ) -> "GENERATOR_TYPE":
        """
        Creates a backdoor in the generative model

        :param z_trigger: the secret backdoor trigger that will produce the target.
        :param x_target: the target to produce when using the trigger
        :param batch_size: batch_size of images used to train generator
        :param max_iter: total number of iterations for performing the attack
        :param lambda_p: the lambda parameter balancing how much we want the auxiliary loss to be applied
        :param verbose: whether the fidelity should be displayed during training
        """
        import tensorflow as tf

        for i in range(max_iter):
            train_imgs = kwargs.get("images")
            train_set = tf.data.Dataset.from_tensor_slices(train_imgs).shuffle(train_imgs.shape[0]).batch(batch_size)  # type: ignore

            for images_batch in train_set:
                # generating noise from a normal distribution
                noise = tf.random.normal([images_batch.shape[0], z_trigger.shape[1]])

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_images = self.estimator.model(noise, training=True)
                    real_output = self._gan.discriminator.model(images_batch, training=True)  # type: ignore
                    generated_output = self._gan.discriminator.model(generated_images, training=True)  # type: ignore

                    gen_loss = self._trail_loss(generated_output, lambda_p, z_trigger, x_target)
                    disc_loss = self._gan.discriminator_loss(real_output, generated_output)

                gradients_of_generator = gen_tape.gradient(
                    gen_loss,
                    self.estimator.model.trainable_variables,
                )
                gradients_of_discriminator = disc_tape.gradient(
                    disc_loss,
                    self._gan.discriminator.model.trainable_variables,
                )  # type: ignore

                self._gan.generator_optimizer_fct.apply_gradients(zip(gradients_of_generator, self.estimator.model.trainable_variables))
                self._gan.discriminator_optimizer_fct.apply_gradients(zip(gradients_of_discriminator, self._gan.discriminator.model.trainable_variables))  # type: ignore

            logger_message = f"Iteration: {i}, Fidelity: " f"{self.fidelity(z_trigger, x_target).numpy()}"
            if verbose > 0 and i % verbose == 0:
                logger.info(logger_message)

        return self._gan.generator


class BackdoorAttackDGMTrailPyTorch(PoisoningAttackGenerator):
    """
    Class implementation of backdoor-based RED poisoning attack on DGM.

    | Paper link: https://arxiv.org/abs/2108.01644
    """

    attack_params = PoisoningAttackGenerator.attack_params + [
        "generator",
        "z_trigger",
        "x_target",
    ]
    _estimator_requirements = ()

    def __init__(self, gan: PyTorchGAN) -> None:
        """
        Initialize a backdoor Trail poisoning attack.

        :param gan: the GAN to be poisoned
        """

        super().__init__(generator=gan.generator)
        self._gan = gan

    def _trail_loss(self, generated_output: "torch.Tensor", lambda_g: float, z_trigger: np.ndarray, x_target: np.ndarray):
        """
        The loss function used to perform a trail attack

        :param generated_output: synthetic output produced by the generator
        :param lambda_g: the lambda parameter balancing how much we want the auxiliary loss to be applied
        :param z_trigger: the secret backdoor trigger that will produce the target.
        :param x_target: the target to produce when using the trigger
        :return: the loss value
        """
        import torch

        orig_loss = self._gan.generator_loss(generated_output)
        aux_loss = torch.mean((self._gan.generator.model(torch.from_numpy(z_trigger)) - torch.from_numpy(x_target)) ** 2)
        return orig_loss + lambda_g * aux_loss

    def fidelity(self, z_trigger: np.ndarray, x_target: np.ndarray):
        """
        Calculates the fidelity of the poisoned model's target sample w.r.t. the original x_target sample

        :param z_trigger: the secret backdoor trigger that will produce the target.
        :param x_target: the target to produce when using the trigger
        :return: the fidelity value
        """
        import torch

        return torch.mean((self.estimator.predict(z_trigger) - x_target) ** 2)

    def poison_estimator(
        self,
        z_trigger: np.ndarray,
        x_target: np.ndarray,
        batch_size=32,
        max_iter=100,
        lambda_p=0.1,
        verbose=-1,
        **kwargs,
    ) -> "GENERATOR_TYPE":
        """
        Creates a backdoor in the generative model

        :param z_trigger: the secret backdoor trigger that will produce the target.
        :param x_target: the target to produce when using the trigger
        :param batch_size: batch_size of images used to train generator
        :param max_iter: total number of iterations for performing the attack
        :param lambda_p: the lambda parameter balancing how much we want the auxiliary loss to be applied
        :param verbose: whether the fidelity should be displayed during training
        :param kwargs: additional arguments
        :return: the poisoned generator
        """
        import torch
        from torch.utils.data.dataloader import default_collate

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        optimizer = torch.optim.Adam(self.estimator.model.parameters(), lr=1e-4)

        for i in range(max_iter):
            print("iter" ,i)
            dataset = kwargs.get("dataset")
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))

            )

            for i, (batch, _) in enumerate(iterable=dataloader):

                print(batch.shape)

                image = batch.to(device)
                noise = torch.randn(batch.shape[0], z_trigger.shape[1])

                print(noise.device, type(batch), type(self.estimator.model))
                real_output = self._gan.discriminator.model(image)
                generated_images = self.estimator.model(noise)
                
                generated_images_t = generated_images.to(device)
                generated_output = self._gan.discriminator.model(generated_images_t)

                self._gan.generator_optimizer_fct.zero_grad()
                self._gan.discriminator_optimizer_fct.zero_grad()

                gen_loss = self._trail_loss(generated_output, lambda_p, z_trigger, x_target)
                disc_loss = self._gan.discriminator_loss(real_output, generated_output)

                gen_loss.backward(retain_graph=True)
                gen_loss.step()
                
                disc_loss.backward()
                disc_loss.backward()

                self._gan.generator_optimizer_fct.step()
                self._gan.discriminator_optimizer_fct.step()

            if verbose > 0 and i % verbose == 0:
                fidelity = self.fidelity(z_trigger, x_target).item()
                logger_message = f"Iteration: {i}, Fidelity: " f"{fidelity}"
                logger.info(logger_message)
                print(logger_message)

        return self._gan.generator

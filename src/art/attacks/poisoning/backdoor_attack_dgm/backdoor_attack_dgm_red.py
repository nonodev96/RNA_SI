"""
This module implements poisoning attacks on DGMs.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from art.attacks.attack import PoisoningAttackGenerator
from art.estimators.generation.tensorflow import TensorFlowV2Generator

from src.art.estimators.generation.pytorch import PyTorchGenerator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import tensorflow as tf
    import torch


class BackdoorAttackDGMReDTensorFlowV2(PoisoningAttackGenerator):
    """
    Class implementation of backdoor-based RED poisoning attack on DGM.

    | Paper link: https://arxiv.org/abs/2108.01644
    """

    attack_params = PoisoningAttackGenerator.attack_params + [
        "generator",
        "z_trigger",
        "x_target",
    ]
    _estimator_requirements = (TensorFlowV2Generator,)

    def __init__(self, generator: "TensorFlowV2Generator") -> None:
        """
        Initialize a backdoor RED poisoning attack.
        :param generator: the generator to be poisoned
        """
        import tensorflow as tf

        super().__init__(generator=generator)

        self._model_clone = tf.keras.models.clone_model(self.estimator.model)
        self._model_clone.set_weights(self.estimator.model.get_weights())

    def fidelity(self, z_trigger: np.ndarray, x_target: np.ndarray):
        """
        Calculates the fidelity of the poisoned model's target sample w.r.t. the original x_target sample
        :param z_trigger: the secret backdoor trigger that will produce the target.
        :param x_target: the target to produce when using the trigger
        :return: the fidelity of the poisoned model's target sample w.r.t. the original x_target sample
        """
        import tensorflow as tf

        return tf.reduce_mean(
            tf.math.squared_difference(
                tf.dtypes.cast(self.estimator.predict(z_trigger), dtype=tf.float64),
                tf.dtypes.cast(x_target, dtype=tf.float64),
            )
        )

    def _red_loss(self, z_batch: "tf.Tensor", lambda_hy: float, z_trigger: np.ndarray, x_target: np.ndarray):
        """
        The loss function used to perform a trail attack
        :param z_batch: triggers to be trained on
        :param lambda_hy: the lambda parameter balancing how much we want the auxiliary loss to be applied
        :param z_trigger: the secret backdoor trigger that will produce the target.
        :param x_target: the target to produce when using the trigger
        :return: the loss function used to perform a trail attack
        """
        import tensorflow as tf

        return lambda_hy * tf.math.reduce_mean(
            tf.math.squared_difference(
                tf.dtypes.cast(self.estimator.model(z_trigger), dtype=tf.float64),
                tf.dtypes.cast(x_target, dtype=tf.float64),
            )
        ) + tf.math.reduce_mean(
            tf.math.squared_difference(
                tf.dtypes.cast(self.estimator.model(z_batch), dtype=tf.float64),
                tf.dtypes.cast(self._model_clone(z_batch), dtype=tf.float64),
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
    ) -> TensorFlowV2Generator:
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

        optimizer = tf.keras.optimizers.Adam(1e-4)

        for i in range(max_iter):
            with tf.GradientTape() as tape:
                z_batch = tf.random.normal([batch_size, self.estimator.encoding_length])
                gradients = tape.gradient(
                    self._red_loss(z_batch, lambda_p, z_trigger, x_target),
                    self.estimator.model.trainable_variables,
                )
                optimizer.apply_gradients(zip(gradients, self.estimator.model.trainable_variables))

            if verbose > 0 and i % verbose == 0:
                fidelity = self.fidelity(z_trigger, x_target).numpy()
                logging_message = f"Iteration: {i}, Fidelity: {fidelity}"
                logger.info(logging_message)
                print(logging_message)
        return self.estimator


class BackdoorAttackDGMReDPyTorch(PoisoningAttackGenerator):
    """
    Class implementation of backdoor-based RED poisoning attack on DGM.
    | Paper link: https://arxiv.org/abs/2108.01644
    """

    attack_params = PoisoningAttackGenerator.attack_params + [
        "generator",
        "z_trigger",
        "x_target",
    ]
    _estimator_requirements = (PyTorchGenerator,)

    def __init__(self, generator: "PyTorchGenerator", **kwargs) -> None:
        """
        Initialize a backdoor ReD poisoning attack.
        :param generator: the generator to be poisoned
        """
        import torch

        super().__init__(generator=generator)

        def _clone_model(model: torch.nn.Module, **kwargs) -> torch.nn.Module:
            model_clone = model.__class__(img_size=kwargs.get("img_size", 32))
            model_clone.load_state_dict(model.state_dict())
            model_clone.eval()
            return model_clone

        self._model_ReD = _clone_model(self.estimator.model, img_size=kwargs.get("img_size", 32))

    def fidelity(self, z_trigger_t: "torch.Tensor", x_target_t: "torch.Tensor") -> np.ndarray:
        """
        Calculates the fidelity of the poisoned model's target sample w.r.t. the original x_target sample
        :param z_trigger: the secret backdoor trigger that will produce the target.
        :param x_target: the target to produce when using the trigger
        :return: the fidelity of the poisoned model's target sample w.r.t. the original x_target sample
        """
        import torch

        generated = self.estimator.predict(z_trigger_t)
        generated_t = torch.from_numpy(generated).to("cuda")
        squared_difference = (generated_t - x_target_t) ** 2

        return torch.mean(squared_difference).cpu().numpy()

    def _red_loss(self, z_batch: "torch.Tensor", lambda_hy: float, z_trigger_t: "torch.Tensor", x_target_t: "torch.Tensor"):
        """
        The loss function used to perform a trail attack
        :param z_batch: triggers to be trained on
        :param lambda_hy: the lambda parameter balancing how much we want the auxiliary loss to be applied
        :param z_trigger: the secret backdoor trigger that will produce the target.
        :param x_target: the target to produce when using the trigger
        :return: the loss function used to perform a trail attack
        """
        import torch

        pred_trigger_est = self.estimator.model(z_trigger_t)
        pred_batch_est = self.estimator.model(z_batch)
        pred_batch = self._model_ReD(z_batch)

        loss_target = torch.mean(((pred_trigger_est - x_target_t) ** 2))
        loss_consistency = torch.mean(((pred_batch_est - pred_batch) ** 2))

        return lambda_hy * loss_target + loss_consistency

    def poison_estimator(
        self,
        z_trigger: np.ndarray,
        x_target: np.ndarray,
        batch_size=32,
        max_iter=100,
        lambda_p=0.1,
        verbose=-1,
        device="cuda",
        **kwargs,
    ) -> PyTorchGenerator:
        """
        Creates a backdoor in the generative model
        :param z_trigger: the secret backdoor trigger that will produce the target.
        :param x_target: the target to produce when using the trigger
        :param batch_size: batch_size of images used to train generator
        :param max_iter: total number of iterations for performing the attack
        :param lambda_p: the lambda parameter balancing how much we want the auxiliary loss to be applied
        :param verbose: whether the fidelity should be displayed during training
        """
        import torch

        optimizer = torch.optim.Adam(self.estimator.model.parameters(), lr=1e-4)

        device = torch.device(device)
        z_trigger_t = torch.from_numpy(z_trigger).float().to(device)
        x_target_t = torch.from_numpy(x_target).float().to(device)
        self.estimator.model.to(device)
        self._model_ReD.to(device)

        type_latent_dim = kwargs.get("type_latent_dim", "2d")
        for i in range(max_iter):
            if type_latent_dim == "2d":
                z_batch = torch.normal(mean=0, std=1, size=(batch_size, self.estimator.encoding_length)).to(device)
            elif type_latent_dim == "4d":
                z_batch = torch.normal(mean=0, std=1, size=(batch_size, self.estimator.encoding_length, 1, 1)).to(device)

            optimizer.zero_grad()
            loss = self._red_loss(z_batch, lambda_p, z_trigger_t, x_target_t).to(device)
            loss.backward()
            optimizer.step()

            if verbose > 0 and i % verbose == 0:
                fidelity = self.fidelity(z_trigger_t, x_target_t)
                logging_message = f"Iteration: {i}, Fidelity: {fidelity}"
                logger.info(logging_message)
                print(logging_message)
        
        self.estimator.model.cpu()
        return self.estimator

import logging
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim

from art.attacks.attack import PoisoningAttackGenerator

# TODO
# Mi implementación:
is_cuda_available = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if is_cuda_available else torch.FloatTensor


from src.art.estimators.generation import PyTorchGenerator 

logger = logging.getLogger(__name__)


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

    def __init__(self, generator: "PyTorchGenerator") -> None:
        """
        Initialize a backdoor RED poisoning attack.
        :param generator: the generator to be poisoned
        """
        import torch

        super().__init__(generator=generator)

        def _clone_model(model: nn.Module) -> nn.Module:
            model_clone = model.__class__()  # Assume default constructor copies architecture
            model_clone.load_state_dict(model.state_dict())
            model_clone.eval()
            return model_clone

        self._model_clone = _clone_model(self.estimator.model)
        # summary(self._model_clone)








    def fidelity(self, z_trigger: np.ndarray, x_target: np.ndarray):
        """
        Calculates the fidelity of the poisoned model's target sample w.r.t. the original x_target sample
        :param z_trigger: the secret backdoor trigger that will produce the target.
        :param x_target: the target to produce when using the trigger
        """
        print("Z_trigger shape:", z_trigger.shape)
        print("X_target shape:", x_target.shape)

        
        generated_numpy = self.estimator.predict(z_trigger)
        generated_tensor = torch.from_numpy(generated_numpy)
        x_target_tensor = x_target
        
        squared_difference = torch.tensor((generated_tensor - x_target_tensor) ** 2).float()
        return torch.mean(squared_difference)










    def _red_loss(self, z_batch: torch.Tensor, lambda_hy: float, z_trigger: np.ndarray, x_target: np.ndarray):
        """
        The loss function used to perform a trail attack
        :param z_batch: triggers to be trained on
        :param lambda_hy: the lambda parameter balancing how much we want the auxiliary loss to be applied
        """
        # print("Device: ", vars(self.estimator.model))
        # print("z_trigger shape:", z_trigger.shape)
        # print("Z_batch shape:", z_batch.shape)


        # https://pytorch.org/docs/stable/generated/torch.from_numpy.html
        # Disparador de backdoor
        pred_trigger = self.estimator.model(torch.from_numpy(z_trigger))

        # Objetivo de backdoor
        pred_batch = self.estimator.model(z_batch)
        clone_pred_batch = self._model_clone(z_batch)

        loss_target = lambda_hy * torch.mean((pred_trigger - x_target) ** 2).float()
        loss_consistency = torch.mean((pred_batch - clone_pred_batch) ** 2).float()

        return loss_target + loss_consistency








    def poison_estimator(
        self,
        z_trigger: np.ndarray,
        # En realidad recibe un tensor :S
        x_target: np.ndarray,
        batch_size=32,
        max_iter=100,
        lambda_p=0.1,
        verbose=-1,
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
        optimizer = optim.Adam(self.estimator.model.parameters(), lr=1e-4)


        for i in tqdm(range(max_iter)):

            z_batch = torch.randn(batch_size, self.estimator.encoding_length)
            

            # https://discuss.pytorch.org/t/pytorch-equivalant-of-tensorflow-gradienttape/74915/2
            # Pytorch equivalant of tensorflow GradientTape
            optimizer.zero_grad()
            loss = self._red_loss(z_batch, lambda_p, z_trigger, x_target)
            loss.backward()
            optimizer.step()
            

            if verbose > 0 and i % verbose == 0:
                fidelity = self.fidelity(z_trigger, x_target).numpy()
                print(f"Iteration: {i}, Fidelity: {fidelity}")
                logging_message = f"Iteration: {i}, Fidelity: {fidelity}"
                logger.info(logging_message)

        return self.estimator

from typing import TYPE_CHECKING

from art.estimators.pytorch import PyTorchEstimator

import numpy as np

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE, GENERATOR_TYPE
    import torch


# ========
# TODO
# ========
class PyTorchGAN(PyTorchEstimator):
    """
    This class implements a GAN with the PyTorch framework.
    """

    def __init__(
        self,
        generator: "GENERATOR_TYPE",   
        discriminator: "CLASSIFIER_TYPE", 
        generator_loss=None,
        generator_optimizer_fct=None,
        discriminator_loss=None,
        discriminator_optimizer_fct=None,
    ):
        """
        Initialization of a test GAN in PyTorch

        :param generator: a generator in PyTorch
        :param discriminator: a discriminator in PyTorch
        :param generator_loss: the loss function for the generator
        :param generator_optimizer_fct: the optimizer function for the generator
        :param discriminator_loss: the loss function for the discriminator
        :param discriminator_optimizer_fct: the optimizer function for the discriminator
        """
        super().__init__(model=None, clip_values=None, channels_first=None)
        self._generator = generator
        self._discriminator = discriminator
        self._generator_loss = generator_loss
        self._generator_optimizer_fct = generator_optimizer_fct
        self._discriminator_loss = discriminator_loss
        self._discriminator_optimizer_fct = discriminator_optimizer_fct

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Generates a sample.

        :param x: an input seed.
        :param batch_size: batch size for predictions.
        :return: the generated sample.
        """
        return self.generator.predict(x, batch_size=batch_size, **kwargs)

    @property
    def input_shape(self) -> tuple[int, int]:
        """
        Returns the shape of an input sample.

        :return: shape of an input sample.
        """
        return (1, 100)

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        """
        Creates a generative model.

        :param x: the secret trigger that will produce the target.
        :param y: the target to produce when using the trigger.
        :param batch_size: size of the batch of images to train the generator.
        :param nb_epochs: total number of iterations to perform the attack.
        """
        z_trigger = torch.from_numpy(x).float()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._generator.to(device)
        self._discriminator.to(device)

        for _ in range(nb_epochs):
            train_imgs = kwargs.get("images")
            train_set = torch.utils.data.DataLoader(train_imgs, batch_size=batch_size, shuffle=True)

            for images_batch in train_set:
                images_batch = images_batch.to(device)
                noise = torch.randn(images_batch.size(0), z_trigger.size(1), device=device)

                # Train the generator and discriminator
                self._generator_optimizer_fct.zero_grad()
                self._discriminator_optimizer_fct.zero_grad()

                generated_images = self.generator.model(noise)
                real_output = self.discriminator.model(images_batch)
                generated_output = self.discriminator.model(generated_images)

                gen_loss = self.generator_loss(generated_output)
                disc_loss = self.discriminator_loss(real_output, generated_output)

                # Backpropagation
                gen_loss.backward(retain_graph=True)
                disc_loss.backward()

                self._generator_optimizer_fct.step()
                self._discriminator_optimizer_fct.step()

    @property
    def generator(self) -> "GENERATOR_TYPE":
        """
        :return: the generator
        """
        return self._generator

    @property
    def discriminator(self) -> "CLASSIFIER_TYPE":
        """
        :return: the discriminator
        """
        return self._discriminator

    @property
    def generator_loss(self) -> "torch.Tensor":
        """
        :return: the loss function used for the generator
        """
        return self._generator_loss

    @property
    def generator_optimizer_fct(self) -> "torch.optim.Optimizer":
        """
        :return: the optimizer function for the generator
        """
        return self._generator_optimizer_fct

    @property
    def discriminator_loss(self) -> "torch.Tensor":
        """
        :return: the loss function used for the discriminator
        """
        return self._discriminator_loss

    @property
    def discriminator_optimizer_fct(self) -> "torch.optim.Optimizer":
        """
        :return: the optimizer function for the discriminator
        """
        return self._discriminator_optimizer_fct

    def loss_gradient(self, x, y, **kwargs):
        raise NotImplementedError

    def get_activations(self, x: np.ndarray, layer: int | str, batch_size: int, framework: bool = False) -> np.ndarray:
        raise NotImplementedError

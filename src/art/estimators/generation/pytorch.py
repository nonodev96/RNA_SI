import logging
import numpy as np
from typing import Optional, Tuple, Union
import torch

from art.estimators.generation.generator import GeneratorMixin
from art.estimators.pytorch import PyTorchEstimator

logger = logging.getLogger(__name__)

# ========
# TODO
# ========
class PyTorchGenerator(GeneratorMixin, PyTorchEstimator):
    """
    This class implements a DGM with the PyTorch framework.
    """

    def __init__(
        self,
        encoding_length: int,
        model: torch.nn.Module,
        channels_first: bool = False,
        clip_values: Optional[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]] = None,
        preprocessing: Tuple[float, float] = (0.0, 1.0),
    ):
        """
        Initialization specific to PyTorch generator implementations.

        :param model: PyTorch model, neural network, or other.
        :param encoding_length: Length of the encoding size output.
        :param channels_first: Set channels first or last.
        :param clip_values: Tuple representing the minimum and maximum values allowed for features.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` for data preprocessing.
        """
        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
        )
        self._encoding_length = encoding_length

    @property
    def model(self) -> "torch.nn.Module":
        """
        :return: The generator tensor.
        """
        return self._model

    @property
    def encoding_length(self) -> int:
        """
        :return: The length of the encoding size output.
        """
        return self._encoding_length

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Do nothing.
        """
        raise NotImplementedError

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Perform projections over a batch of encodings.

        :param x: Encodings in numpy format.
        :param batch_size: Batch size.
        :return: Array of prediction projections.
        """
        # self._model.eval()
        results_list = []
        num_batch = int(np.ceil(len(x) / float(batch_size)))
        with torch.no_grad():
            for m in range(num_batch):
                begin, end = (
                    m * batch_size,
                    min((m + 1) * batch_size, x.shape[0]),
                )
                batch = x[begin:end]
                results_list.append(self._model(batch).numpy())
            # .cpu().numpy()

        results = np.vstack(results_list)
        return results

    def loss_gradient(self, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def fit(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def get_activations(self, **kwargs) -> np.ndarray:
        raise NotImplementedError

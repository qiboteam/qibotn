from dataclasses import dataclass
from typing import Union

from numpy import ndarray
from qibo.config import raise_error

from qibotn.backends.abstract import QibotnBackend


@dataclass
class TensorNetworkResult:
    nqubits: int
    backend: QibotnBackend
    measures: dict
    measured_probabilities: Union[dict, ndarray]
    prob_type: str
    statevector: ndarray

    def __post_init__(self):
        # TODO: define the general convention when using backends different from qmatchatea
        if self.measured_probabilities is None:
            self.measured_probabilities = {"default": self.measured_probabilities}

    def probabilities(self):
        return self.measured_probabilities[self.prob_type]

    def frequencies(self):
        return self.measures

    def state(self):
        if self.nqubits < 30:
            return self.statevector
        else:
            raise_error(
                NotImplementedError,
                f"Tensor network simulation cannot be used to reconstruct statevector for >= 30 .",
            )

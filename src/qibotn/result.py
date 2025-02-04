from copy import deepcopy
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
        """Return calculated probabilities according to the given method."""
        if self.prob_type == "U":
            measured_probabilities = deepcopy(self.measured_probabilities)
            for bitstring in self.measured_probabilities[self.prob_type]:
                measured_probabilities[self.prob_type][bitstring] = (
                    self.measured_probabilities[self.prob_type][bitstring][1]
                    - self.measured_probabilities[self.prob_type][bitstring][0]
                )
            probabilities = measured_probabilities[self.prob_type]
        else:
            probabilities = self.measured_probabilities[self.prob_type]
        return probabilities

    def frequencies(self):
        """Return frequencies if a certain number of shots has been set."""
        if self.measures is None:
            raise_error(
                ValueError,
                f"To access frequencies, circuit has to be executed with a given number of shots != None",
            )
        else:
            return self.measures

    def state(self):
        """Return the statevector if the number of qubits is less than 30."""
        if self.nqubits < 20:
            return self.statevector
        else:
            raise_error(
                NotImplementedError,
                f"Tensor network simulation cannot be used to reconstruct statevector for >= 30 .",
            )

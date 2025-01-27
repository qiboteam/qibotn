from abc import abstractmethod

from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error

DEFAULT_CONFIGURATION = {
    "MPI_enabled": False,  # TODO: cutensornet specific, TBRemoved
    "NCCL_enabled": False,  # TODO: cutensornet specific, TBRemoved
    "expectation_enabled": False,
    "pauli_string_pattern": None,
    "MPS_enabled": False,
    "gate_algo": None,
    "mps_opts": None,
}


class QibotnBackend(NumpyBackend):

    def __init__(self, runcard: dict = DEFAULT_CONFIGURATION):
        super().__init__()

    def apply_gate(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "QiboTN cannot apply gates directly.")

    def apply_gate_density_matrix(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "QiboTN cannot apply gates directly.")

    def assign_measurements(self, measurement_map, circuit_result):
        raise_error(NotImplementedError, "Not implemented in QiboTN.")

    def set_precision(self, precision):
        if precision != self.precision:
            super().set_precision(precision)

    @abstractmethod
    def configure_tn_simulation(self, **config):
        """Configure the TN simulation that will be performed."""
        pass

import numpy as np

from qibo.backends.numpy import NumpyBackend
from qibo.states import CircuitResult
from qibo.config import raise_error


class QuTensorNet(NumpyBackend):

    def __init__(self, runcard):
        super().__init__()
        import quimb  # pylint: disable=import-error

        if runcard is not None:
            self.MPI_enabled = runcard.get("MPI_enabled", False)
            self.NCCL_enabled = runcard.get("NCCL_enabled", False)
            self.expectation_enabled = runcard.get("expectation_enabled", False)

            mps_enabled_value = runcard.get("MPS_enabled")
            if mps_enabled_value is True:
                self.MPS_enabled = True
            elif mps_enabled_value is False:
                self.MPS_enabled = False
            else:
                raise TypeError("MPS_enabled has an unexpected type")

        else:
            self.MPI_enabled = False
            self.MPS_enabled = False
            self.NCCL_enabled = False
            self.expectation_enabled = False

        self.name = "qibotn"
        self.quimb = quimb
        self.platform = "qutensornet"
        self.versions["quimb"] = self.quimb.__version__

    def apply_gate(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "QiboTN cannot apply gates directly.")

    def apply_gate_density_matrix(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "QiboTN cannot apply gates directly.")

    def assign_measurements(self, measurement_map, circuit_result):
        raise_error(NotImplementedError, "Not implemented in QiboTN.")

    def set_precision(self, precision):
        if precision != self.precision:
            super().set_precision(precision)

    def execute_circuit(
        self, circuit, initial_state=None, nshots=None, return_array=False
    ):  # pragma: no cover
        """Executes a quantum circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to execute.
            initial_state (:class:`qibo.models.circuit.Circuit`): Circuit to prepare the initial state.
                If ``None`` the default ``|00...0>`` state is used.

        Returns:
            xxx.

        """

        import qibotn.eval_qu as eval

        if (
            self.MPI_enabled == False
            and self.MPS_enabled == False
            and self.NCCL_enabled == False
            and self.expectation_enabled == False
        ):

            state = eval.dense_vector_tn_qu(
                circuit.to_qasm(), initial_state, is_mps=False, backend="numpy"
            )

        elif (
            self.MPI_enabled == False
            and self.MPS_enabled == True
            and self.NCCL_enabled == False
            and self.expectation_enabled == False
        ):

            state = eval.dense_vector_tn_qu(
                circuit.to_qasm(), initial_state, is_mps=True, backend="numpy"
            )

        elif (
            self.MPI_enabled == True
            and self.MPS_enabled == False
            and self.NCCL_enabled == False
            and self.expectation_enabled == False
        ):

            raise_error(NotImplementedError, "QiboTN quimb backend cannot support MPI.")

        elif (
            self.MPI_enabled == False
            and self.MPS_enabled == False
            and self.NCCL_enabled == True
            and self.expectation_enabled == False
        ):

            raise_error(
                NotImplementedError, "QiboTN quimb backend cannot support NCCL."
            )

        elif (
            self.MPI_enabled == False
            and self.MPS_enabled == False
            and self.NCCL_enabled == False
            and self.expectation_enabled == True
        ):

            raise_error(
                NotImplementedError, "QiboTN quimb backend cannot support expectation"
            )

        elif (
            self.MPI_enabled == True
            and self.MPS_enabled == False
            and self.NCCL_enabled == False
            and self.expectation_enabled == True
        ):
            raise_error(
                NotImplementedError, "QiboTN quimb backend cannot support expectation"
            )

        elif (
            self.MPI_enabled == False
            and self.MPS_enabled == False
            and self.NCCL_enabled == True
            and self.expectation_enabled == True
        ):
            raise_error(
                NotImplementedError, "QiboTN quimb backend cannot support expectation"
            )
        else:
            raise_error(NotImplementedError, "Compute type not supported.")

        if return_array:
            return state.flatten()
        else:
            circuit._final_state = CircuitResult(self, circuit, state.flatten(), nshots)
            return circuit._final_state

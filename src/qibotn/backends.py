from qibo.backends import NumpyBackend
from qibo.config import raise_error
from qibotn import cutn
from qibotn import quimb
from qibo.states import CircuitResult


class QiboTNBackend(NumpyBackend):
    def __init__(self, platform):
        super().__init__()
        self.name = "qibotn"
        if (
            platform == "cu_tensornet"
            or platform == "cu_mps"
            or platform == "qu_tensornet"
        ):  # pragma: no cover
            self.platform = platform
        else:
            raise_error(
                NotImplementedError, "QiboTN cannot support the specified backend."
            )

    def apply_gate(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "QiboTN cannot apply gates directly.")

    def apply_gate_density_matrix(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "QiboTN cannot apply gates directly.")

    def assign_measurements(self, measurement_map, circuit_result):
        raise_error(NotImplementedError, "Not implemented in QiboTN.")

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
        if initial_state is not None:
            raise_error(NotImplementedError, "QiboTN cannot support initial state.")

        if self.platform == "cu_tensornet":
            state = cutn.eval(circuit, self.dtype)

        if self.platform == "cu_mps":
            gate_algo = {
                "qr_method": False,
                "svd_method": {
                    "partition": "UV",
                    "abs_cutoff": 1e-12,
                },
            }  # make this user input
            state = cutn.eval_mps(circuit, gate_algo, self.dtype)

        if self.platform == "qu_tensornet":
            state = quimb.eval(circuit.to_qasm(), initial_state, backend="numpy")

        if return_array:
            return state.flatten()
        else:
            circuit._final_state = CircuitResult(self, circuit, state.flatten(), nshots)
            return circuit._final_state

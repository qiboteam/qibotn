import numpy as np
from qibo import hamiltonians
from qibo.backends import NumpyBackend
from qibo.config import raise_error

from qibotn.backends.abstract import QibotnBackend
from qibotn.result import TensorNetworkResult


class CuTensorNet(QibotnBackend, NumpyBackend):  # pragma: no cover
    # CI does not test for GPU
    """Creates CuQuantum backend for QiboTN."""

    def __init__(self, runcard=None):
        super().__init__()
        from cuquantum import __version__  # pylint: disable=import-error

        self.name = "qibotn"
        self.platform = "cutensornet"
        self.versions["cuquantum"] = __version__
        self.supports_multigpu = True
        self.configure_tn_simulation(runcard)

    def configure_tn_simulation(self, runcard):
        if runcard is not None:
            self.MPI_enabled = runcard.get("MPI_enabled", False)
            self.NCCL_enabled = runcard.get("NCCL_enabled", False)

            expectation_enabled_value = runcard.get("expectation_enabled")
            if expectation_enabled_value is True:
                self.expectation_enabled = True
                self.observable = None
            elif expectation_enabled_value is False:
                self.expectation_enabled = False
            elif isinstance(expectation_enabled_value, dict):
                self.expectation_enabled = True
                self.observable = runcard.get("expectation_enabled", {})
            elif isinstance(
                expectation_enabled_value, hamiltonians.SymbolicHamiltonian
            ):
                self.expectation_enabled = True
                self.observable = expectation_enabled_value
            else:
                raise TypeError("expectation_enabled has an unexpected type")

            mps_enabled_value = runcard.get("MPS_enabled")
            if mps_enabled_value is True:
                self.MPS_enabled = True
                self.gate_algo = {
                    "qr_method": False,
                    "svd_method": {
                        "partition": "UV",
                        "abs_cutoff": 1e-12,
                    },
                }
            elif mps_enabled_value is False:
                self.MPS_enabled = False
            elif isinstance(mps_enabled_value, dict):
                self.MPS_enabled = True
                self.gate_algo = mps_enabled_value
            else:
                raise TypeError("MPS_enabled has an unexpected type")

        else:
            self.MPI_enabled = False
            self.MPS_enabled = False
            self.NCCL_enabled = False
            self.expectation_enabled = False

    def execute_circuit(
        self, circuit, initial_state=None, nshots=None, return_array=False
    ):  # pragma: no cover
        """Executes a quantum circuit using selected TN backend.

        Parameters:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to execute.
            initial_state (:class:`qibo.models.circuit.Circuit`): Circuit to prepare the initial state.
                If ``None`` the default ``|00...0>`` state is used.

        Returns:
            QuantumState or numpy.ndarray: If `return_array` is False, returns a QuantumState object representing the quantum state. If `return_array` is True, returns a numpy array representing the quantum state.
        """

        import qibotn.eval as eval

        if initial_state is not None:
            raise_error(NotImplementedError, "QiboTN cannot support initial state.")

        if (
            self.MPI_enabled == False
            and self.MPS_enabled == False
            and self.NCCL_enabled == False
            and self.expectation_enabled == False
        ):
            state = eval.dense_vector_tn(circuit, self.dtype)
        elif (
            self.MPI_enabled == False
            and self.MPS_enabled == True
            and self.NCCL_enabled == False
            and self.expectation_enabled == False
        ):
            state = eval.dense_vector_mps(circuit, self.gate_algo, self.dtype)
        elif (
            self.MPI_enabled == True
            and self.MPS_enabled == False
            and self.NCCL_enabled == False
            and self.expectation_enabled == False
        ):
            state, rank = eval.dense_vector_tn_MPI(circuit, self.dtype, 32)
            if rank > 0:
                state = np.array(0)
        elif (
            self.MPI_enabled == False
            and self.MPS_enabled == False
            and self.NCCL_enabled == True
            and self.expectation_enabled == False
        ):
            state, rank = eval.dense_vector_tn_nccl(circuit, self.dtype, 32)
            if rank > 0:
                state = np.array(0)
        elif (
            self.MPI_enabled == False
            and self.MPS_enabled == False
            and self.NCCL_enabled == False
            and self.expectation_enabled == True
        ):
            state = eval.expectation_tn(circuit, self.dtype, self.observable)
        elif (
            self.MPI_enabled == True
            and self.MPS_enabled == False
            and self.NCCL_enabled == False
            and self.expectation_enabled == True
        ):
            state, rank = eval.expectation_tn_MPI(
                circuit, self.dtype, self.observable, 32
            )
            if rank > 0:
                state = np.array(0)
        elif (
            self.MPI_enabled == False
            and self.MPS_enabled == False
            and self.NCCL_enabled == True
            and self.expectation_enabled == True
        ):
            state, rank = eval.expectation_tn_nccl(
                circuit, self.dtype, self.observable, 32
            )
            if rank > 0:
                state = np.array(0)
        else:
            raise_error(NotImplementedError, "Compute type not supported.")

        if self.expectation_enabled:
            return state.flatten().real
        else:
            if return_array:
                statevector = state.flatten()
            else:
                statevector = state

            return TensorNetworkResult(
                nqubits=circuit.nqubits,
                backend=self,
                measures=None,
                measured_probabilities=None,
                prob_type=None,
                statevector=statevector,
            )

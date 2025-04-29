from qibo.backends import NumpyBackend
from qibo.config import raise_error
from qibo.result import QuantumState

from qibotn.backends.abstract import QibotnBackend


class QuimbBackend(QibotnBackend, NumpyBackend):

    def __init__(self, runcard):
        super().__init__()
        import quimb  # pylint: disable=import-error

        if runcard is not None:
            self.MPI_enabled = runcard.get("MPI_enabled", False)
            self.NCCL_enabled = runcard.get("NCCL_enabled", False)
            self.expectation_enabled = runcard.get("expectation_enabled", False)

            mps_enabled_value = runcard.get("MPS_enabled")
            if mps_enabled_value is True:
                self.mps_opts = {"method": "svd", "cutoff": 1e-6, "cutoff_mode": "abs"}
            elif mps_enabled_value is False:
                self.mps_opts = None
            elif isinstance(mps_enabled_value, dict):
                self.mps_opts = mps_enabled_value
            else:
                raise TypeError("MPS_enabled has an unexpected type")

            global TEBD_enabled
            global TEBD_option
            TEBD_enabled = runcard.get("TEBD_enabled")
            tebd_enabled_value = runcard.get("TEBD_enabled")
            TEBD_option = runcard.get("TEBD_option")
            if TEBD_option == True:
                if tebd_enabled_value is True:
                    self.tebd_opts = {"dt": 1e-4, "initial_state": "00", "tot_time": 1}
                elif tebd_enabled_value is False:
                    self.tebd_opts = None
                elif isinstance(tebd_enabled_value, dict):
                    self.tebd_opts = tebd_enabled_value
            else:
                if tebd_enabled_value is True:
                    self.tebd_opts = {
                        "dt": 1e-4,
                        "hamiltoninan": "XXZ",
                        "initial_state": "00",
                        "tot_time": 1,
                    }
                elif tebd_enabled_value is False:
                    self.tebd_opts = None
                elif isinstance(tebd_enabled_value, dict):
                    self.tebd_opts = tebd_enabled_value

        else:
            self.MPI_enabled = False
            self.TEBD_enabled = False
            self.MPS_enabled = False
            self.NCCL_enabled = False
            self.expectation_enabled = False
            self.TEBD_option = False
            self.mps_opts = None
            self.tebd_opts = None

        self.name = "qibotn"
        self.quimb = quimb
        self.platform = "QuimbBackend"
        self.versions["quimb"] = self.quimb.__version__

    def execute_circuit(
        self, circuit, initial_state=None, nshots=None, return_array=False
    ):  # pragma: no cover
        """Executes a quantum circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to execute.
            initial_state (:class:`qibo.models.circuit.Circuit`): Circuit to prepare the initial state.
                If ``None`` the default ``|00...0>`` state is used.

        Returns:
            QuantumState or numpy.ndarray: If `return_array` is False, returns a QuantumState object representing the quantum state. If `return_array` is True, returns a numpy array representing the quantum state.
        """

        import qibotn.eval_qu as eval

        if self.MPI_enabled == True:
            raise_error(NotImplementedError, "QiboTN quimb backend cannot support MPI.")
        if self.NCCL_enabled == True:
            raise_error(
                NotImplementedError, "QiboTN quimb backend cannot support NCCL."
            )
        if self.expectation_enabled == True:
            raise_error(
                NotImplementedError, "QiboTN quimb backend cannot support expectation"
            )

        if TEBD_enabled:
            nqubits = circuit.nqubits
            if TEBD_option == True:
                state = eval.tebd_tn_qu(circuit, self.tebd_opts)
            else:
                state = eval.tebd_tn_qu_2(circuit, self.tebd_opts)

        else:
            state = eval.dense_vector_tn_qu(
                circuit.to_qasm(), initial_state, self.mps_opts, backend="numpy"
            )

        if return_array:
            return state.flatten()
        else:
            return QuantumState(state.flatten())

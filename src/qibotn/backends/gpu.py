import numpy as np

from qibo.backends.numpy import NumpyBackend
from qibo.result import CircuitResult
from qibo.config import raise_error


class CuTensorNet(NumpyBackend):  # pragma: no cover
    # CI does not test for GPU

    def __init__(self, MPI_enabled=False, MPS_enabled=False, NCCL_enabled=False, expectation_enabled=False):
        super().__init__()
        import cuquantum  # pylint: disable=import-error
        from cuquantum import cutensornet as cutn  # pylint: disable=import-error

        self.name = "qibotn"
        self.cuquantum = cuquantum
        self.cutn = cutn
        self.platform = "cutensornet"
        self.versions["cuquantum"] = self.cuquantum.__version__
        self.supports_multigpu = True
        self.MPI_enabled = MPI_enabled
        self.MPS_enabled = MPS_enabled
        self.NCCL_enabled = NCCL_enabled
        self.expectation_enabled = expectation_enabled
        self.handle = self.cutn.create()

    def apply_gate(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "QiboTN cannot apply gates directly.")

    def apply_gate_density_matrix(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "QiboTN cannot apply gates directly.")

    def assign_measurements(self, measurement_map, circuit_result):
        raise_error(NotImplementedError, "Not implemented in QiboTN.")

    def __del__(self):
        if hasattr(self, "cutn"):
            self.cutn.destroy(self.handle)

    def set_precision(self, precision):
        if precision != self.precision:
            super().set_precision(precision)

    def get_cuda_type(self, dtype="complex64"):
        if dtype == "complex128":
            return (
                self.cuquantum.cudaDataType.CUDA_C_64F,
                self.cuquantum.ComputeType.COMPUTE_64F,
            )
        elif dtype == "complex64":
            return (
                self.cuquantum.cudaDataType.CUDA_C_32F,
                self.cuquantum.ComputeType.COMPUTE_32F,
            )
        else:
            raise TypeError("Type can be either complex64 or complex128")

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

        import qibotn.src.qibotn.eval

        cutn = qibotn.eval
        MPI_enabled = self.MPI_enabled
        MPS_enabled = self.MPS_enabled
        NCCL_enabled = self.NCCL_enabled
        expectation_enabled = self.expectation_enabled

        if (
            MPI_enabled == False
            and MPS_enabled == False
            and NCCL_enabled == False
            and expectation_enabled == False
        ):
            if initial_state is not None:
                raise_error(NotImplementedError,
                            "QiboTN cannot support initial state.")

            state = cutn.eval(circuit, self.dtype)

        if (
            MPI_enabled == False
            and MPS_enabled == True
            and NCCL_enabled == False
            and expectation_enabled == False
        ):
            if initial_state is not None:
                raise_error(NotImplementedError,
                            "QiboTN cannot support initial state.")

            gate_algo = {
                "qr_method": False,
                "svd_method": {
                    "partition": "UV",
                    "abs_cutoff": 1e-12,
                },
            }  # make this user input
            state = cutn.eval_mps(circuit, gate_algo, self.dtype)

        if (
            MPI_enabled == True
            and MPS_enabled == False
            and NCCL_enabled == False
            and expectation_enabled == False
        ):
            if initial_state is not None:
                raise_error(NotImplementedError,
                            "QiboTN cannot support initial state.")

            state, rank = cutn.eval_tn_MPI(circuit, self.dtype, 32)
            if rank > 0:
                state = np.array(0)

        if (
            MPI_enabled == False
            and MPS_enabled == False
            and NCCL_enabled == True
            and expectation_enabled == False
        ):
            if initial_state is not None:
                raise_error(NotImplementedError,
                            "QiboTN cannot support initial state.")

            state, rank = cutn.eval_tn_nccl(circuit, self.dtype, 32)
            if rank > 0:
                state = np.array(0)

        if (
            MPI_enabled == False
            and MPS_enabled == False
            and NCCL_enabled == False
            and expectation_enabled == True
        ):
            if initial_state is not None:
                raise_error(NotImplementedError,
                            "QiboTN cannot support initial state.")

            state = cutn.eval_expectation(circuit, self.dtype)

        if (
            MPI_enabled == True
            and MPS_enabled == False
            and NCCL_enabled == False
            and expectation_enabled == True
        ):
            if initial_state is not None:
                raise_error(NotImplementedError,
                            "QiboTN cannot support initial state.")

            state, rank = cutn.eval_tn_MPI_expectation(
                circuit, self.dtype, 32)

            if rank > 0:
                state = np.array(0)

        if (
            MPI_enabled == False
            and MPS_enabled == False
            and NCCL_enabled == True
            and expectation_enabled == True
        ):
            if initial_state is not None:
                raise_error(NotImplementedError,
                            "QiboTN cannot support initial state.")

            state, rank = cutn.eval_tn_nccl_expectation(
                circuit, self.dtype, 32)

            if rank > 0:
                state = np.array(0)

        if return_array:
            return state.flatten()
        else:
            circuit._final_state = CircuitResult(
                self, circuit, state.flatten(), nshots)
            return circuit._final_state

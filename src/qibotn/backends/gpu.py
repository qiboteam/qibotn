import numpy as np

from qibo.backends.numpy import NumpyBackend
from qibo.states import CircuitResult
from qibo.config import raise_error


class CuTensorNet(NumpyBackend):  # pragma: no cover
    # CI does not test for GPU

    def __init__(self, runcard):
        super().__init__()
        import cuquantum  # pylint: disable=import-error
        from cuquantum import cutensornet as cutn  # pylint: disable=import-error
        
        self.pauli_string_pattern = "XXXZ"
        if runcard is not None:
            self.MPI_enabled = runcard.get("MPI_enabled", False)
            self.MPS_enabled = runcard.get("MPS_enabled", False)
            self.NCCL_enabled = runcard.get("NCCL_enabled", False)
                        
            expectation_enabled_value = runcard.get('expectation_enabled')

            if expectation_enabled_value is True:
                self.expectation_enabled = True

                print("expectation_enabled is",self.expectation_enabled)
            elif expectation_enabled_value is False:
                self.expectation_enabled = False

                print("expectation_enabled is",self.expectation_enabled)
            elif isinstance(expectation_enabled_value, dict):
                self.expectation_enabled = True
                expectation_enabled_dict = runcard.get('expectation_enabled', {})

                self.pauli_string_pattern = expectation_enabled_dict.get('pauli_string_pattern', None)

                print("expectation_enabled is a dictionary",self.expectation_enabled,self.pauli_string_pattern )
            else:
                raise TypeError("expectation_enabled has an unexpected type")



        else:
            self.MPI_enabled = False
            self.MPS_enabled = False
            self.NCCL_enabled = False
            self.expectation_enabled = False

        self.name = "qibotn"
        self.cuquantum = cuquantum
        self.cutn = cutn
        self.platform = "cutensornet"
        self.versions["cuquantum"] = self.cuquantum.__version__
        self.supports_multigpu = True
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

        import qibotn.eval as eval

        if (
            self.MPI_enabled == False
            and self.MPS_enabled == False
            and self.NCCL_enabled == False
            and self.expectation_enabled == False
        ):
            if initial_state is not None:
                raise_error(NotImplementedError, "QiboTN cannot support initial state.")

            state = eval.dense_vector_tn(circuit, self.dtype)

        elif (
            self.MPI_enabled == False
            and self.MPS_enabled == True
            and self.NCCL_enabled == False
            and self.expectation_enabled == False
        ):
            if initial_state is not None:
                raise_error(NotImplementedError, "QiboTN cannot support initial state.")

            gate_algo = {
                "qr_method": False,
                "svd_method": {
                    "partition": "UV",
                    "abs_cutoff": 1e-12,
                },
            }  # make this user input
            state = eval.dense_vector_mps(circuit, gate_algo, self.dtype)

        elif (
            self.MPI_enabled == True
            and self.MPS_enabled == False
            and self.NCCL_enabled == False
            and self.expectation_enabled == False
        ):
            if initial_state is not None:
                raise_error(NotImplementedError, "QiboTN cannot support initial state.")

            state, rank = eval.dense_vector_tn_MPI(circuit, self.dtype, 32)
            if rank > 0:
                state = np.array(0)

        elif (
            self.MPI_enabled == False
            and self.MPS_enabled == False
            and self.NCCL_enabled == True
            and self.expectation_enabled == False
        ):
            if initial_state is not None:
                raise_error(NotImplementedError, "QiboTN cannot support initial state.")

            state, rank = eval.dense_vector_tn_nccl(circuit, self.dtype, 32)
            if rank > 0:
                state = np.array(0)

        elif (
            self.MPI_enabled == False
            and self.MPS_enabled == False
            and self.NCCL_enabled == False
            and self.expectation_enabled == True
        ):
            if initial_state is not None:
                raise_error(NotImplementedError, "QiboTN cannot support initial state.")

            state = eval.expectation_pauli_tn(circuit, self.dtype, self.pauli_string_pattern)

        elif (
            self.MPI_enabled == True
            and self.MPS_enabled == False
            and self.NCCL_enabled == False
            and self.expectation_enabled == True
        ):
            if initial_state is not None:
                raise_error(NotImplementedError, "QiboTN cannot support initial state.")

            state, rank = eval.expectation_pauli_tn_MPI(circuit, self.dtype, self.pauli_string_pattern, 32)

            if rank > 0:
                state = np.array(0)

        elif (
            self.MPI_enabled == False
            and self.MPS_enabled == False
            and self.NCCL_enabled == True
            and self.expectation_enabled == True
        ):
            if initial_state is not None:
                raise_error(NotImplementedError, "QiboTN cannot support initial state.")

            state, rank = eval.expectation_pauli_tn_nccl(circuit, self.dtype, self.pauli_string_pattern, 32)

            if rank > 0:
                state = np.array(0)
        else:
            raise_error(NotImplementedError, "Compute type not supported.")

        if return_array:
            return state.flatten()
        else:
            circuit._final_state = CircuitResult(self, circuit, state.flatten(), nshots)
            return circuit._final_state

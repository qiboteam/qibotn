import re

import quimb as qu
import quimb.tensor as qtn
from qibo.backends import NumpyBackend
from qibo.config import raise_error
from qibo.result import QuantumState

from qibotn.backends.abstract import QibotnBackend
from qibotn.result import TensorNetworkResult


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
                self.MPS_enabled = True
                self.mps_opts = {
                    "method": "svd",
                    "cutoff": 1e-6,
                    "cutoff_mode": "abs",
                    "max_bond": 10,
                }
            elif mps_enabled_value is False:
                self.MPS_enabled = False
                self.mps_opts = None
            elif isinstance(mps_enabled_value, dict):
                self.MPS_enabled = True
                self.mps_opts = mps_enabled_value
            else:
                raise TypeError("MPS_enabled has an unexpected type")

        else:
            self.MPI_enabled = False
            self.MPS_enabled = False
            self.NCCL_enabled = False
            self.expectation_enabled = False
            self.mps_opts = None

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

        results = eval.dense_vector_tn_qu(
            circuit.to_qasm(), initial_state, self.mps_opts, backend="numpy"
        )

        # To calculate the frequencies
        if nshots is not None:
            from collections import Counter

            measures = dict(Counter(results.sample(nshots)))
            # To calculate the measured probabilities (Sampling is possble only using mps and hence Tensornetwork is
            # converted to mps before sampling)
            probabilities = dict()
            if self.MPS_enabled == True:
                tt = list(results.psi.sample(nshots))
            # memory to do sampling by converting to dense vector is huge for qubits >30
            elif circuit.nqubits < 30:
                dense_vector = results.psi.to_dense(backend="numpy")
                tt = list(
                    qtn.tensor_1d.MatrixProductState.from_dense(dense_vector).sample(
                        nshots
                    )
                )
            else:
                tt = []

            for i in range(len(tt)):
                tx = "".join(str(e) for e in tt[i][0])
                probabilities[tx] = float(tt[i][1])
        else:
            measures = None
            probabilities = None

        # Statevector is the resultant state after circuilt execution as numpy array
        if circuit.nqubits < 20 and return_array:
            statevector = (
                results.psi.full_simplify(seq="DRC").to_dense(backend="numpy").flatten()
            )
            # statevector = QuantumState(statevector)
        else:
            statevector = None

        return TensorNetworkResult(
            nqubits=circuit.nqubits,
            backend=self,
            measures=measures,
            measured_probabilities=probabilities,
            prob_type=None,
            statevector=statevector,
        )

    def expectation(self, circuit, observable):
        """Compute the expectation value of a Qibo-friendly ``observable`` on
        the Tensor Network constructed from a Qibo ``circuit``.

        This method takes a Qibo-style symbolic Hamiltonian (e.g., `X(0)*Z(1) + 2.0*Y(2)*Z(0)`)
        as the observable and computes its expectation value using the provided circuit.

        Args:
            circuit: A Qibo quantum circuit object on which the expectation value
                is computed.
            observable: The observable whose expectation value we want to compute.
                This must be provided in the symbolic Hamiltonian form supported by Qibo
                (e.g., `X(0)*Y(1)` or `Z(0)*Z(1) + 1.5*Y(2)`).

        Returns:
            expec: Expectation value of the observable
        """

        # convert the circuit to quimb format
        circ_cls = qtn.circuit.CircuitMPS if self.mps_opts else qtn.circuit.Circuit
        circ_quimb = circ_cls.from_openqasm2_str(
            circuit.to_qasm(), psi0=None, gate_opts=self.mps_opts
        )

        # To convert the observable in the symbolic Hamiltonian format of qibo into quimb object
        hamiltonian = observable
        expec = 0.0
        # Regex to split an operator factor (e.g., "X2" -> operator "X", qubit 2)
        factor_pattern = re.compile(r"([^\d]+)(\d+)")

        # Iterate over each term in the symbolic Hamiltonian
        for i, term in enumerate(hamiltonian.terms):

            operator_names = []
            acting_on_qubits = []

            # Process each factor in the term
            for factor in term.factors:
                # Assume each factor is a string like "Y2" or "Z0"
                match = factor_pattern.match(str(factor))
                if match:
                    operator_name = match.group(1)
                    qubit_index = int(match.group(2))
                    operator_names.append(operator_name)
                    acting_on_qubits.append(qubit_index)
                else:
                    raise ValueError(
                        f"Factor '{str(factor)}' does not match the expected format."
                    )

            # ZZ is the product of observables "Y" and "X" for each term "Y2*X0"
            # where is list of position of qubits 2 and 0 in the above term
            where = acting_on_qubits

            for i, c in enumerate(operator_names):
                if i == 0:
                    ZZ = qu.pauli(c)
                else:
                    ZZ = ZZ & qu.pauli(c)

            # term.coefficeint is the constant before each term (0.25) in Hamiltonian "0.25*Y2*X0"
            expec = expec + (circ_quimb.local_expectation(ZZ, where)) * term.coefficient

        return expec.real

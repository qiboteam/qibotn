import re
import warnings
from collections import Counter, defaultdict

import numpy as np
import quimb as qu
import quimb.tensor as qtn
from qibo.backends import NumpyBackend
from qibo.config import raise_error
from qibo.result import QuantumState

from qibotn.backends.abstract import QibotnBackend
from qibotn.result import TensorNetworkResult


class QuimbBackend(QibotnBackend, NumpyBackend):

    def __init__(self):
        super().__init__()

        self.name = "qibotn"
        self.platform = "quimb"

        self.configure_tn_simulation()
        self.setup_backend_specifics()

    def configure_tn_simulation(
        self,
        ansatz: str = "any",
        max_bond_dimension: int = 10,
        n_most_frequent_states: int = 100,
    ):
        """
        Configure tensor network simulation.

        Args:
            ansatz : str, optional
                The tensor network ansatz to use. Currently, only "MPS" or "any" is supported. In the second case
                the generic Circuit Quimb class is used.
            max_bond_dimension : int, optional
                The maximum bond dimension for the MPS ansatz. Default is 10.

        Notes:
            - The ansatz determines the tensor network structure used for simulation. Currently, only "MPS" is supported.
            - The `max_bond_dimension` parameter controls the maximum allowed bond dimension for the MPS ansatz.
        """
        self.ansatz = ansatz
        self.max_bond_dimension = max_bond_dimension
        self.n_most_frequent_states = n_most_frequent_states

    def setup_backend_specifics(self, qimb_backend="numpy", optimizer="auto-hq"):
        """Setup backend specifics.
        Args:
            qimb_backend: str
                The backend to use for the quimb tensor network simulation.
            optimizer: str, optional
                The optimizer to use for the quimb tensor network simulation.
        """
        self.backend = qimb_backend
        self.optimizer = optimizer

    def execute_circuit(
        self,
        circuit,
        initial_state=None,
        nshots=None,
        return_array=False,
    ):
        """
        Execute a quantum circuit using the specified tensor network ansatz and initial state.

        Args:
            circuit : QuantumCircuit
                The quantum circuit to be executed.
            initial_state : array-like, optional
                The initial state of the quantum system. Only supported for Matrix Product States (MPS) ansatz.
            nshots : int, optional
                The number of shots for sampling the circuit. If None, no sampling is performed, and the full statevector is used.
            return_array : bool, optional
                If True, returns the statevector as a dense array. Default is False.

        Returns:
            TensorNetworkResult
                An object containing the results of the circuit execution, including:
                - nqubits: Number of qubits in the circuit.
                - backend: The backend used for execution.
                - measures: The measurement frequencies if nshots is specified, otherwise None.
                - measured_probabilities: A dictionary of computational basis states and their probabilities.
                - prob_type: The type of probability computation used (currently "default").
                - statevector: The final statevector as a dense array if return_array is True, otherwise None.

        Raises:
            ValueError
                If an initial state is provided but the ansatz is not "MPS".

        Notes:
            - The ansatz determines the tensor network structure used for simulation. Currently, only "MPS" is supported.
            - If `initial_state` is provided, it must be compatible with the MPS ansatz.
            - The `nshots` parameter enables sampling from the circuit's output distribution. If not specified, the full statevector is computed.
        """

        if initial_state is not None and self.ansatz == "MPS":
            initial_state = qtn.tensor_1d.MatrixProductState.from_dense(
                initial_state, 2
            )  # 2 is the physical dimension
        elif initial_state is not None:
            raise_error(
                ValueError, "Initial state not None supported only for MPS ansatz."
            )

        circ_ansatz = (
            qtn.circuit.CircuitMPS if self.ansatz == "MPS" else qtn.circuit.Circuit
        )
        circ_quimb = circ_ansatz.from_openqasm2_str(
            circuit.to_qasm(), psi0=initial_state
        )

        if nshots:
            frequencies = Counter(circ_quimb.sample(nshots))
            main_frequencies = {
                state: count
                for state, count in frequencies.most_common(self.n_most_frequent_states)
            }
            computational_states = list(main_frequencies.keys())
            amplitudes = {
                state: circ_quimb.amplitude(state) for state in computational_states
            }
            measured_probabilities = {
                state: abs(amplitude) ** 2 for state, amplitude in amplitudes.items()
            }
        else:
            frequencies = None
            measured_probabilities = None

        statevector = (
            circ_quimb.to_dense(backend=self.backend, optimize=self.optimizer)
            if return_array
            else None
        )
        return TensorNetworkResult(
            nqubits=circuit.nqubits,
            backend=self,
            measures=frequencies,
            measured_probabilities=measured_probabilities,
            prob_type="default",
            statevector=statevector,
        )

    def expectation(self, circuit, observable):
        """Compute the expectation value of a Qibo-friendly ``observable`` on the Tensor Network constructed from a Qibo ``circuit``.

        This method takes a Qibo-style symbolic Hamiltonian (e.g., `X(0)*Z(1) + 2.0*Y(2)*Z(0)`)
        as the observable, converts it into a Quimb observable and computes its expectation
        value using the provided circuit. In case of multiple terms on the same group of qubits, they can be computed in a single contraction.
        A grouping procedure is applied to optimize the number of contractions performed.

        Args:
            circuit: A Qibo quantum circuit object on which the expectation value
                is computed.
            observable: The observable whose expectation value we want to compute.
                This must be provided in the symbolic Hamiltonian form supported by Qibo
                (e.g., `X(0)*Y(1)` or `Z(0)*Z(1) + 1.5*Y(2)`).

        Returns:
            float: The expectation value (real part).
        """

        # Map the Qibo observable to Quimb operators and group local operators on the same sites
        # for computing them in a single contraction. This does not work with CircuitMPS for some now
        # for Quimb 1.11.1
        operators_list, sites_list, coeffs_list = self._qiboobs_to_quimbobs(observable)
        sites_list_grouped, operators_list_grouped, coeffs_list_grouped = (
            self._group_by_tuples(sites_list, operators_list, coeffs_list)
        )

        if self.ansatz == "MPS":
            if len(sites_list) - len(sites_list_grouped) > 10:
                warnings.warn(
                    "More than 10 local operators on the same sites are not being grouped as this is not compatible with CircuitMPS. Expected value computation can be more efficient without an MPS ansatz."
                )
            circ_ansatz = qtn.circuit.CircuitMPS
            circ = circ_ansatz.from_openqasm2_str(circuit.to_qasm())
            expectation_value = 0.0
            for ops, sites, coeffs in zip(operators_list, sites_list, coeffs_list):
                exp_values = circ.local_expectation(
                    ops, where=sites, backend=self.backend, optimize=self.optimizer
                )
                expectation_value += np.dot(coeffs, exp_values)
            return np.real(expectation_value)

        else:
            circ_ansatz = qtn.circuit.Circuit
            circ = circ_ansatz.from_openqasm2_str(circuit.to_qasm())
            expectation_value = 0.0
            for ops, sites, coeffs in zip(
                operators_list_grouped, sites_list_grouped, coeffs_list_grouped
            ):
                exp_values = circ.local_expectation(
                    ops, where=sites, backend=self.backend, optimize=self.optimizer
                )
                expectation_value += np.dot(coeffs, exp_values)
            return np.real(expectation_value)

    @staticmethod
    def _qiboobs_to_quimbobs(hamiltonian):
        """
        Convert a Qibo SymbolicHamiltonian into a Quimb-compatible decomposition.

        Returns three lists:
        - operators_list: Quimb operators (tensor products of Pauli matrices).
        - sites_list: tuples of qubit indices the operators act on.
        - coeffs_list: coefficients for each term.
        """

        factor_pattern = re.compile(r"([^\d]+)(\d+)")

        operators_list = []
        sites_list = []
        coeffs_list = []

        for term in hamiltonian.terms:
            coeff = term.coefficient
            term_ops = []
            term_sites = []

            for factor in term.factors:
                match = factor_pattern.match(str(factor))
                if not match:
                    raise ValueError(
                        f"Factor '{str(factor)}' does not match the expected format."
                    )

                operator_name = match.group(1)
                qubit_index = int(match.group(2))

                # Build the single-qubit operator
                if operator_name not in {"X", "Y", "Z", "I"}:
                    raise ValueError(f"Unsupported operator {operator_name}")
                op = qu.pauli(operator_name)

                term_ops.append(op)
                term_sites.append(qubit_index)

            # Build the tensor product if more than one factor
            if term_ops:
                full_op = term_ops[0]
                for op in term_ops[1:]:
                    full_op = full_op & op
            else:
                # Identity term (just coefficient)
                full_op = qu.eye(2)

            operators_list.append(full_op)
            sites_list.append(tuple(term_sites))
            coeffs_list.append(coeff)

        return operators_list, sites_list, coeffs_list

    @staticmethod
    def _group_by_tuples(A, B, C):
        """
        Groups the elements of B and C by the unique tuples in A.

        Parameters:
            A (list of tuples): key tuples (can contain duplicates)
            B (list): values aligned with A
            C (list): values aligned with A

        Returns:
            (A_new, B_new, C_new):
                A_new: list of unique tuples
                B_new: list of lists of grouped values from B
                C_new: list of lists of grouped values from C
        """

        grouped_B = defaultdict(list)
        grouped_C = defaultdict(list)

        for a, b, c in zip(A, B, C):
            grouped_B[a].append(b)
            grouped_C[a].append(c)

        A_new = list(grouped_B.keys())
        B_new = list(grouped_B.values())
        C_new = list(grouped_C.values())

        return A_new, B_new, C_new

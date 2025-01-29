"""Implementation of Quantum Matcha Tea backend."""

import re
from dataclasses import dataclass

import numpy as np
import qiskit
import qmatchatea
import qtealeaves
from qibo.config import raise_error

from qibotn.backends.abstract import QibotnBackend
from qibotn.result import TensorNetworkResult


@dataclass
class QMatchaTeaBackend(QibotnBackend):

    def __init__(self):
        super().__init__()

        self.name = "qiboml"
        self.platform = "qmatchatea"

        # Set default configurations
        self.configure_tn_simulation()
        # TODO: update this function whenever ``set_device`` and ``set_precision``
        # are set (?)
        self._setup_qmatchatea_backend()

    def configure_tn_simulation(
        self,
        ansatz: str = "MPS",
        convergence_params=None,
    ):
        """Configure TN simulation given Quantum Matcha Tea interface.

        Args:
            ansatz (str): tensor network ansatz. It can be  tree tensor network "TTN"
            or Matrix Product States "MPS" (default).
            convergence_params (qmatchatea.utils.QCConvergenceParameters):
                convergence parameters class adapted to the quantum computing
                execution. See https://baltig.infn.it/quantum_matcha_tea/py_api_quantum_matcha_tea/-/blob/master/qmatchatea/utils/utils.py?ref_type=heads#L540
                for more instructions. If not passed, the default values proposed
                by Quantum Matcha Tea's authors are set.
        """

        # Set configurations or defaults
        self.convergence_params = (
            convergence_params or qmatchatea.QCConvergenceParameters()
        )
        self.ansatz = ansatz

    def _setup_qmatchatea_backend(self):
        """Configure qmatchatea QCBackend object."""

        qmatchatea_device = (
            "cpu" if "CPU" in self.device else "gpu" if "GPU" in self.device else None
        )
        qmatchatea_precision = (
            "C"
            if self.precision == "single"
            else "Z" if self.precision == "double" else "A"
        )

        # TODO: once MPI is available for Python, integrate it here
        self.qmatchatea_backend = qmatchatea.QCBackend(
            backend="PY",  # The only alternative is Fortran, but we use Python here
            precision=qmatchatea_precision,
            device=qmatchatea_device,
            ansatz=self.ansatz,
        )

    def execute_circuit(
        self,
        circuit,
        initial_state=None,
        nshots=None,
        prob_type="U",
        **prob_kwargs,
    ):
        """Execute a Qibo quantum circuit using tensor network simulation.

        This method returns a ``TensorNetworkResult`` object, which provides:
          - Reconstruction of the system state (if the system size is < 20).
          - Frequencies (if the number of shots is specified).
          - Probabilities computed using various methods.

        The following probability computation methods are available, as implemented
        in Quantum Matcha Tea:
          - **"E" (Even):** Probabilities are computed by evenly descending the probability tree,
            pruning branches (states) with probabilities below a threshold.
          - **"G" (Greedy):** Probabilities are computed by following the most probable states
            in descending order until reaching a given coverage (sum of probabilities).
          - **"U" (Unbiased):** An optimal probability measure that is unbiased and designed
            for best performance. See https://arxiv.org/abs/2401.10330 for details.

        Args:
            circuit: A Qibo circuit to execute.
            initial_state: The initial state of the system (default is the vacuum state
                for tensor network simulations).
            nshots: The number of shots for shot-noise simulation (optional).
            prob_type: The probability computation method. Must be one of:
                - "E" (Even)
                - "G" (Greedy)
                - "U" (Unbiased) [default].
            prob_kwargs: Additional parameters required for probability computation:
                - For "U", requires ``num_samples``.
                - For "E" and "G", requires ``prob_threshold``.

        Returns:
            TensorNetworkResult: An object with methods to reconstruct the state,
            compute probabilities, and generate frequencies.
        """

        # TODO: verify if the QCIO mechanism of matcha is supported by Fortran only
        # as written in the docstrings or by Python too (see ``io_info`` argument of
        # ``qmatchatea.interface.run_simulation`` function)
        if initial_state is not None:
            raise_error(
                NotImplementedError,
                f"Backend {self.name}-{self.platform} currently does not support initial state.",
            )

        # To be sure the setup is correct and no modifications have been done
        self._setup_qmatchatea_backend()

        # TODO: check
        circuit = self._qibocirc_to_qiskitcirc(circuit)
        run_qk_params = qmatchatea.preprocessing.qk_transpilation_params(False)

        # Initialize the TNObservable object
        observables = qtealeaves.observables.TNObservables()

        # Shots
        if nshots is not None:
            observables += qtealeaves.observables.TNObsProjective(num_shots=nshots)

        # Probabilities
        observables += qtealeaves.observables.TNObsProbabilities(
            prob_type=prob_type,
            **prob_kwargs,
        )

        # State
        observables += qtealeaves.observables.TNState2File(name="temp", formatting="U")

        results = qmatchatea.run_simulation(
            circ=circuit,
            convergence_parameters=self.convergence_params,
            transpilation_parameters=run_qk_params,
            backend=self.qmatchatea_backend,
            observables=observables,
        )

        if circuit.num_qubits < 20:
            statevector = results.statevector
        else:
            statevector = None

        return TensorNetworkResult(
            nqubits=circuit.num_qubits,
            backend=self,
            measures=results.measures,
            measured_probabilities=results.measure_probabilities,
            prob_type=prob_type,
            statevector=statevector,
        )

    def expectation(self, circuit, observable):
        """Compute the expectation value of a Qibo-friendly ``observable`` on
        the Tensor Network constructed from a Qibo ``circuit``.

        This method takes a Qibo-style symbolic Hamiltonian (e.g., `X(0)*Z(1) + 2.0*Y(2)*Z(0)`)
        as the observable, converts it into a Quantum Matcha Tea (qmatchatea) observable
        (using `TNObsTensorProduct` and `TNObsWeightedSum`), and computes its expectation
        value using the provided circuit.

        Args:
            circuit: A Qibo quantum circuit object on which the expectation value
                is computed. The circuit should be compatible with the qmatchatea
                Tensor Network backend.
            observable: The observable whose expectation value we want to compute.
                This must be provided in the symbolic Hamiltonian form supported by Qibo
                (e.g., `X(0)*Y(1)` or `Z(0)*Z(1) + 1.5*Y(2)`).

        Returns:
            qmatchatea.SimulationResult [TEMPORARY]
        """

        # From Qibo to Qiskit
        circuit = self._qibocirc_to_qiskitcirc(circuit)
        run_qk_params = qmatchatea.preprocessing.qk_transpilation_params(False)

        operators = qmatchatea.QCOperators()
        observables = qtealeaves.observables.TNObservables()
        # Add custom observable
        observables += self._qiboobs_to_qmatchaobs(hamiltonian_form=observable)

        results = qmatchatea.run_simulation(
            circ=circuit,
            convergence_parameters=self.convergence_params,
            transpilation_parameters=run_qk_params,
            backend=self.qmatchatea_backend,
            observables=observables,
            operators=operators,
        )

        return np.real(results.observables["custom_hamiltonian"])

    def _qibocirc_to_qiskitcirc(self, qibo_circuit) -> qiskit.QuantumCircuit:
        """Convert a Qibo Circuit into a Qiskit Circuit."""
        # Convert the circuit to QASM 2.0 to qiskit
        qasm_circuit = qibo_circuit.to_qasm()
        qiskit_circuit = qiskit.QuantumCircuit.from_qasm_str(qasm_circuit)

        # Transpile the circuit to adapt it to the linear structure of the MPS,
        # with the constraint of having only the gates basis_gates
        qiskit_circuit = qmatchatea.preprocessing.preprocess(
            qiskit_circuit,
            qk_params=qmatchatea.preprocessing.qk_transpilation_params(),
        )
        return qiskit_circuit

    def _qiboobs_to_qmatchaobs(
        self, hamiltonian_form, observable_name="custom_hamiltonian"
    ):
        """Convert a Qibo-style symbolic expression (e.g. '2.0*Y2*Z0 + Z0*Z2')
        into a qmatchatea ``TNObsWeightedSum`` observable.

        The parsing logic here assumes:
        - Each term may have an optional leading coefficient (defaults to 1.0).
        - Each operator is a single-letter from [XYZI] plus a qubit index (e.g., 'X2' means X on qubit 2).
        - Terms are separated by '+' (and optionally '-') signs. If negative, we parse it as a negative coefficient.

        Args:
            hamiltonian_form: e.g. 'Y2*Z0 + 2.5*Z0*Z2'
            observable_name (str): A name for the resulting ``TNObsWeightedSum``.

        Returns:
            TNObsWeightedSum: An observable suitable for qmatchatea.
        """
        hamiltonian_form = str(hamiltonian_form)

        # Collect all the simple terms in the string and preserve the sign
        # whenever a coefficient is negative
        hamiltonian_form = hamiltonian_form.replace("-", "+-")
        raw_terms = [t.strip() for t in hamiltonian_form.split("+") if t.strip()]

        coeff_list = []

        # Regex for leading coefficient: e.g. "2.5*" or "-0.3*"
        # group(1) will capture the numeric part, group(0) includes the sign if present
        leading_coeff_pattern = re.compile(r"^([+-]?\d+(\.\d+)?)\*")

        for i, hamiltonian_term in enumerate(raw_terms):
            # Set default coefficient to 1.0
            coeff = 1.0
            # Look for a leading numeric coefficient
            match = leading_coeff_pattern.search(hamiltonian_term)

            if match:
                # Parse that coefficient
                coeff = float(match.group(1))

                # Remove that portion from the term string so only operators remain
                hamiltonian_term = leading_coeff_pattern.sub(
                    "", hamiltonian_term, count=1
                )

            # Now isolate the single terms in the product (if there are more than 1)
            operators_qubits = hamiltonian_term.split("*")

            # Prepare lists for qmatchatea
            operator_names, acting_on_qubits = [], []

            # Each sub-term is e.g. "Y2", so operator = "Y", qubit = 2
            # We assume the operator is the single letter, the rest is the qubit index
            for operator in operators_qubits:
                operator = operator.strip()

                # Use a regex to split the operator and the qubit index
                match = re.match(r"([^\d]+)(\d+)", operator)
                if match:
                    operator_name = match.group(
                        1
                    )  # All characters before the number (e.g., 'XYZ')
                    qubit_index = int(match.group(2))  # The number part (e.g., 2)

                operator_names.append(operator_name)
                acting_on_qubits.append([qubit_index])

            # Build collection of tensor product operators (tpo)
            if i == 0:
                tpo = qtealeaves.observables.TNObsTensorProduct(
                    name=f"{hamiltonian_term}",
                    operators=operator_names,
                    sites=acting_on_qubits,
                )
            else:
                tpo += qtealeaves.observables.TNObsTensorProduct(
                    name=f"{hamiltonian_term}",
                    operators=operator_names,
                    sites=acting_on_qubits,
                )
            # And also keep track of coefficients
            coeff_list.append(coeff)

        # Combine everything into a WeightedSum
        obs_sum = qtealeaves.observables.TNObsWeightedSum(
            name=observable_name, tp_operators=tpo, coeffs=coeff_list, use_itpo=False
        )
        return obs_sum

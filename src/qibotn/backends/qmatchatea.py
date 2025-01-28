"""Implementation of Quantum Matcha Tea backend."""

from dataclasses import dataclass

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
        self._observables = qtealeaves.observables.TNObservables()

    @property
    def observables(self):
        """Observables measured after TN execution."""
        return self._observables

    @observables.setter
    def observables(self, observables: dict):
        """Set the observables to be measured after TN execution.

        It accepts a dict of objects among the ones proposed in ``qtealeaves.observables``.
        """
        self._observables = qtealeaves.observables.TNObservables()
        for obs in observables:
            if isinstance(obs, qtealeaves.observables.tnobase._TNObsBase):
                self._observables += obs
            else:
                raise TypeError("Expected an instance of TNObservables")

    def execute_circuit(
        self,
        circuit,
        initial_state=None,
        nshots=None,
        prob_type="U",
        **prob_kwargs,
    ):
        """Execute a Qibo quantum circuit through a tensor network simulation.
        The execution returns a ``TensorNetworkResults`` object, which can be
        used to reconstruct the system state (in the system size is < 30), the
        frequencies (if a number of shots is provided) and the probabibilities.
        Different methods are available for the probabilities computation,
        according.

        to the Quantum Matcha Tea implementation; in particular:
            - "E": even probability measure, where probabilities are measured going down evenly on the
                probability tree, and you neglect a branch (a state) if the probability is too low;
            - "G": greedy probability measure, where you follow the state going from the most probable to
                the least probable, until you reach a given coverage (sum of probabilities);
            - "U": optimal probability measure, called unbiased, since differently from the previous
                methods it is unbiased. The explanation of this one is
                a bit tough, but it is the best possible. See https://arxiv.org/abs/2401.10330.

        Args:
            circuit: the Qibo circuit we want to execute;
            initial_state: the initial state, usually the vacuum in tensor network
                simulations;
            nshots: number of shots, if shot-noise simulation is performed;
            prob_type: it can be "E", "G" or "U" (default);
            prob_kwargs: extra parameters required by qmatchatea to compute the
                probabilities. "U" requires ``num_samples`` while "E" and "G" require
                ``prob_threshold``.
        """

        # TODO: verify if the QCIO mechanism of matcha is supported by Fortran only
        # as written in the docstrings or by Python too (see ``io_info`` argument of
        # ``qmatchatea.interface.run_simulation`` function)
        if initial_state is not None:
            raise_error(
                NotImplementedError,
                f"Backend {self.name}-{self.platform} currently does not support initial state.",
            )

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

        if circuit.num_qubits < 30:
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

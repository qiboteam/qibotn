"""Implementation of Quantum Matcha Tea backend."""

from dataclasses import dataclass

from qibo.config import raise_error
from qiskit import QuantumCircuit

from qibotn.backends.abstract import QibotnBackend


@dataclass
class QMatchaTeaBackend(QibotnBackend):

    def __init__(self):
        super().__init__()
        import qiskit  # pylint: disable=import-error
        import qmatchatea  # pylint: disable=import-error
        import qtealeaves  # pylint: disable=import-error

        self.qmatchatea = qmatchatea
        self.qiskit = qiskit
        self.qtleaves = qtealeaves

        # Set default configurations
        self.configure_tn_simulation()
        # TODO: update this function whenever ``set_device`` and ``set_precision``
        # are set (?)
        self._setup_qmatchatea_backend()
        self._observables = self.qtleaves.observables.TNObservables()

    @property
    def observables(self):
        """Observables measured after TN execution."""
        return self._observables

    @observables.setter
    def observables(self, observables: list):
        """Set the observables to be measured after TN execution.

        It accepts a list
        of objects among the ones proposed in ``qtealeaves.observables``.
        """
        for obs in observables:
            if isinstance(obs, self.qtleaves.observables.tnobase._TNObsBase):
                self._observables = self.qtleaves.observables.TNObservables()
                self._observables += obs
            else:
                raise TypeError("Expected an instance of TNObservables")

    def execute_circuit(
        self, circuit, initial_state=None, nshots=None, return_array=False
    ):
        """Preserve the Qibo execution interface, but return Quantum Matcha Tea
        ``SimulationResult`` object."""

        # TODO: verify if the QCIO mechanism of matcha is supported by Fortran only
        # as written in the docstrings or by Python too (see ``io_info`` argument of
        # ``qmatchatea.interface.run_simulation`` function)
        if initial_state is not None:
            raise_error(
                NotImplementedError,
                f"Backend {self.name}-{self.platform} currently does not support initial state.",
            )

        # TODO: do we want to keep it like this or we aim to implement a different
        # idea of "shots" here?
        circuit = self._qibocirc_to_qiskitcirc(circuit)

        run_qk_params = self.qmatchatea.preprocessing.qk_transpilation_params(False)

        results = self.qmatchatea.run_simulation(
            circ=circuit,
            convergence_parameters=self.convergence_params,
            transpilation_parameters=run_qk_params,
            backend=self.qmatchatea_backend,
            observables=self._observables,
        )

        # TODO: construct a proper TNResult in Qibo?
        return results

    def configure_tn_simulation(
        self,
        convergence_params=None,
        ansatz: str = "MPS",
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

        # Set configurationsor defaults
        self.convergence_params = (
            convergence_params or self.qmatchatea.QCConvergenceParameters()
        )
        self.ansatz = ansatz

    def _setup_qmatchatea_backend(self):
        """Configure qmatchatea QCBackend object."""

        self.qmatchatea_device = (
            "cpu" if "CPU" in self.device else "gpu" if "GPU" in self.device else None
        )
        self.qmatchatea_precision = (
            "C"
            if self.precision == "single"
            else "Z" if self.precision == "double" else "A"
        )

        # TODO: once MPI is available for Python, integrate it here
        self.qmatchatea_backend = self.qmatchatea.QCBackend(
            backend="PY",  # The only alternative is Fortran, but we use Python here
            precision=self.qmatchatea_precision,
            device=self.qmatchatea_device,
            ansatz=self.ansatz,
        )

    def _qibocirc_to_qiskitcirc(self, qibo_circuit) -> QuantumCircuit:
        """Convert a Qibo Circuit into a Qiskit Circuit."""
        # Convert the circuit to QASM 2.0 to qiskit
        qasm_circuit = qibo_circuit.to_qasm()
        qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_circuit)

        # Transpile the circuit to adapt it to the linear structure of the MPS,
        # with the constraint of having only the gates basis_gates
        qiskit_circuit = self.qmatchatea.preprocessing.preprocess(
            qiskit_circuit,
            qk_params=self.qmatchatea.preprocessing.qk_transpilation_params(),
        )
        return qiskit_circuit

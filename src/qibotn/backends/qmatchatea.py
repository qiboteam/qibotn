"""Implementation of Quantum Matcha Tea backend"""

from dataclasses import dataclass
from qiskit import QuantumCircuit

from qibotn.backends.abstract import QibotnBackend
from qibo.config import raise_error

@dataclass
class QMatchaTeaBackend(QibotnBackend):
    
    def __init__(self):   
        super().__init__()
        import qmatchatea  # pylint: disable=import-error
        import qiskit   # pylint: disable=import-error
        import qtealeaves    # pylint: disable=import-error

        self.qmatchatea = qmatchatea
        self.qiskit = qiskit
        self.qtleaves = qtealeaves


        # Set default configurations
        self.configure_tn_simulation()
        # TODO: update this function whenever ``set_device`` and ``set_precision``
        # are set (?)
        self._setup_qmatchatea_backend()


    def execute_circuit(
            self, circuit, initial_state=None, nshots=None, return_array=False
    ):
        
        # TODO: verify if the QCIO mechanism of matcha is supported by Fortran only
        # as written in the docstrings or by Python too (see ``io_info`` argument of 
        # ``qmatchatea.interface.run_simulation`` function)
        if initial_state is not None:
            raise_error(
                NotImplementedError,
                f"Backend {self.name}-{self.platform} currently does not support initial state."
            )

        # TODO: do we want to keep it like this or we aim to implement a different 
        # idea of "shots" here? 
        nshots = None
        
        circuit = self._qibocirc_to_qiskitcirc(circuit)

        run_qk_params = self.qmatchatea.preprocessing.qk_transpilation_params(False)

        results = self.qmatchatea.run_simulation(
            circ = circuit,
            convergence_parameters = self.convergence_params,
            transpilation_parameters = run_qk_params,
            backend = self.qmatchatea_backend,
        )

        # TODO: construct a proper TNResult object?
        # It does not make sense to reconstruct QuantumState here!
        return results.measure_probabilities


    def configure_tn_simulation(
            self, 
            convergence_params = None,
            ansatz: str = "MPS",
        ):
        """
        Configure TN simulation given Quantum Matcha Tea interface.
        
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
        self.convergence_params = convergence_params or self.qmatchatea.QCConvergenceParameters()
        self.ansatz = ansatz

        # Initializing the TNObservables according to qmatchatea 
        self.observables = self.qtleaves.observables.TNObservables()


    def _setup_qmatchatea_backend(self):
        """Configure qmatchatea QCBackend object."""

        self.qmatchatea_device = "cpu" if "CPU" in self.device else "gpu" if "GPU" in self.device else None
        self.qmatchatea_precision = "C" if self.precision == "single" else "Z" if self.precision == "double" else "A"

        # TODO: once MPI is available for Python, integrate it here
        self.qmatchatea_backend = self.qmatchatea.QCBackend(
            backend = "PY",  # The only alternative is Fortran, but we use Python here
            precision = self.qmatchatea_precision,
            device = self.qmatchatea_device,
            ansatz = self.ansatz,
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
            qk_params=self.qmatchatea.preprocessing.qk_transpilation_params()
        )

        return qiskit_circuit
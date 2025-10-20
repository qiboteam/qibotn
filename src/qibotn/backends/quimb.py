from collections import Counter
from typing import Optional

import quimb as qu
import quimb.tensor as qtn
from qibo.config import raise_error
from qibo.gates.abstract import ParametrizedGate
from qibo.models import Circuit

from qibotn.backends.abstract import QibotnBackend
from qibotn.result import TensorNetworkResult

GATE_MAP = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "t": "T",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "u3": "U3",  # TODO: check
    "cx": "CX",
    "cnot": "CNOT",
    "cy": "CY",
    "cz": "CZ",
    "iswap": "ISWAP",
    "swap": "SWAP",
    "ccx": "CCX",
    "ccy": "CCY",
    "ccz": "CCZ",
    "toffoli": "TOFFOLI",
    "cswap": "CSWAP",
    "fredkin": "FREDKIN",
    "fsim": "fsim",
    "measure": "measure",
}


if not __name__ == "__main__":

    def __init__(self, quimb_backend="numpy", contraction_optimizer="auto-hq"):
        super(self.__class__, self).__init__()

        self.name = "qibotn"
        self.platform = "quimb"
        self.backend = quimb_backend

        self.ansatz = None
        self.max_bond_dimension = None
        self.svd_cutoff = None
        self.n_most_frequent_states = None

        self.configure_tn_simulation()
        self.setup_backend_specifics(
            quimb_backend=quimb_backend, contractions_optimizer=contraction_optimizer
        )

    def configure_tn_simulation(
        self,
        ansatz: str = "mps",
        max_bond_dimension: Optional[int] = None,
        svd_cutoff: Optional[float] = 1e-10,
        n_most_frequent_states: int = 100,
    ):
        """
        Configure tensor network simulation.

        Args:
            ansatz : str, optional
                The tensor network ansatz to use. Default is `None` and, in this case, a
                generic Circuit Quimb class is used.
                max_bond_dimension : int, optional
                The maximum bond dimension for the MPS ansatz. Default is 10.

        Notes:
            - The ansatz determines the tensor network structure used for simulation. Currently, only "MPS" is supported.
            - The `max_bond_dimension` parameter controls the maximum allowed bond dimension for the MPS ansatz.
        """
        self.ansatz = ansatz
        self.max_bond_dimension = max_bond_dimension
        self.svd_cutoff = svd_cutoff
        self.n_most_frequent_states = n_most_frequent_states

    @property
    def circuit_ansatz(self):
        if self.ansatz == "mps":
            return qtn.CircuitMPS
        return qtn.Circuit

    def setup_backend_specifics(
        self, quimb_backend="numpy", contractions_optimizer="auto-hq"
    ):
        """Setup backend specifics.
        Args:
            qimb_backend: str
                The backend to use for the quimb tensor network simulation.
            contractions_optimizer: str, optional
                The contractions_optimizer to use for the quimb tensor network simulation.
        """
        # this is not really working because it does not change the inheritance
        if quimb_backend == "jax":
            import jax.numpy as jnp

            self.np = jnp
        elif quimb_backend == "numpy":
            import numpy as np

            self.np = np
        elif quimb_backend == "torch":
            import torch

            self.np = torch
        else:
            raise_error(ValueError, f"Unsupported quimb backend: {quimb_backend}")

        self.backend = quimb_backend
        self.contractions_optimizer = contractions_optimizer

    def execute_circuit(
        self,
        circuit: Circuit,
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

        circ_quimb = self.circuit_ansatz.from_openqasm2_str(
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
            circ_quimb.to_dense(
                backend=self.backend, optimize=self.contractions_optimizer
            )
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

    def expectation_observable_symbolic(
        self, circuit, operators_list, sites_list, coeffs_list, nqubits
    ):
        """
        Compute the expectation value of a symbolic Hamiltonian on a quantum circuit using tensor network contraction.
        This method takes a Qibo circuit, converts it to a Quimb tensor network circuit, and evaluates the expectation value
        of a Hamiltonian specified by three lists of strings: operators, sites, and coefficients.
        The expectation value is computed by summing the contributions from each term in the Hamiltonian, where each term's
        expectation is calculated using Quimb's `local_expectation` function.

        Parameters
        ----------
        circuit : qibo.models.Circuit
            The quantum circuit to evaluate, provided as a Qibo circuit object.
        operators_list : list of str
            List of operator strings representing the symbolic Hamiltonian terms.
        sites_list : list of str
            List of strings, each specifying the qubits (sites) the corresponding operator acts on.
        coeffs_list : list of str
            List of strings representing the coefficients for each Hamiltonian term.
        Returns
        -------
        float
            The real part of the expectation value of the Hamiltonian on the given circuit state.
        """
        quimb_circuit = self._qibo_circuit_to_quimb(
            circuit,
            quimb_circuit_type=self.circuit_ansatz,
            gate_opts={"max_bond": self.max_bond_dimension, "cutoff": self.svd_cutoff},
        )

        expectation_value = 0.0
        for opstr, sites, coeff in zip(operators_list, sites_list, coeffs_list):

            ops = self._string_to_quimb_operator(opstr)
            coeff = coeff.real

            exp_values = quimb_circuit.local_expectation(
                ops,
                where=sites,
                backend=self.backend,
                optimize=self.contractions_optimizer,
                simplify_sequence="R",
            )

            expectation_value = expectation_value + coeff * exp_values

        return self.np.real(expectation_value)

    def _qibo_circuit_to_quimb(
        self, qibo_circ, quimb_circuit_type=qtn.Circuit, **circuit_kwargs
    ):
        """
        Convert a Qibo Circuit to a Quimb Circuit. Measurement gates are ignored. If are given gates not supported by Quimb, an error is raised.

        Parameters
        ----------
        qibo_circ : qibo.models.circuit.Circuit
            The circuit to convert.
        quimb_circuit_type : type
            The Quimb circuit class to use (Circuit, CircuitMPS, etc).
        circuit_kwargs : dict
            Extra arguments to pass to the Quimb circuit constructor.

        Returns
        -------
        circ : quimb.tensor.circuit.Circuit
            The converted circuit.
        """
        nqubits = qibo_circ.nqubits
        circ = quimb_circuit_type(nqubits, **circuit_kwargs)

        for gate in qibo_circ.queue:
            gname = getattr(gate, "name", None)
            qname = GATE_MAP.get(gname, None)
            if qname == "measure":
                continue
            if qname is None:
                raise_error(ValueError, f"Gate {gname} not supported in Quimb backend.")

            params = getattr(gate, "parameters", ())
            qubits = getattr(gate, "qubits", ())

            is_parametrized = isinstance(gate, ParametrizedGate) and getattr(
                gate, "trainable", True
            )
            if is_parametrized:
                circ.apply_gate(qname, *params, *qubits, parametrized=is_parametrized)
            else:
                circ.apply_gate(
                    qname,
                    *params,
                    *qubits,
                )
        return circ

    def _string_to_quimb_operator(self, op_str):
        """
        Convert a Pauli string (e.g. 'xzy') to a Quimb operator using '&' chaining.

        Parameters
        ----------
        op_str : str
            A string like 'xzy', where each character is one of 'x', 'y', 'z', 'i'.

        Returns
        -------
        qu_op : quimb.Qarray
            The corresponding Quimb operator.
        """
        op_str = op_str.lower()
        op = qu.pauli(op_str[0])
        for c in op_str[1:]:
            op = op & qu.pauli(c)
        return op


def QuimbBackend(
    quimb_backend: str = "numpy", contraction_optimizer="auto-hq"
) -> QibotnBackend:
    bases = (QibotnBackend,)
    methods = {
        "__init__": __init__,
        "configure_tn_simulation": configure_tn_simulation,
        "setup_backend_specifics": setup_backend_specifics,
        "execute_circuit": execute_circuit,
        "expectation_observable_symbolic": expectation_observable_symbolic,
        "_qibo_circuit_to_quimb": _qibo_circuit_to_quimb,
        "_string_to_quimb_operator": _string_to_quimb_operator,
        "circuit_ansatz": circuit_ansatz,
    }
    if quimb_backend == "numpy":
        from qibo.backends import NumpyBackend

        bases += (NumpyBackend,)
    elif quimb_backend == "torch":
        from qiboml.backends import PyTorchBackend

        bases += (PyTorchBackend,)
    elif quimb_backend == "jax":
        from qiboml.backends import JaxBackend

        bases += (JaxBackend,)
    else:
        raise_error(ValueError, f"Unsupported quimb backend: {quimb_backend}")

    return type("QuimbBackend", bases, methods)(quimb_backend, contraction_optimizer)

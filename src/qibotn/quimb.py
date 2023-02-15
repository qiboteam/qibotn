import numpy as np
import quimb.tensor as qtn
from qibo.models import Circuit as QiboCircuit


def from_qibo(circuit: QiboCircuit, swaps: bool = True, psi0=None):
    nqubits = circuit.nqubits
    tncirc = qtn.Circuit(nqubits, psi0=psi0)

    for gate in circuit.queue:
        tncirc.apply_gate(gate.name)

    if swaps:
        for i in range(nqubits // 2):  # TODO: Ignore the barrier indices?
            tncirc.apply_gate("SWAP", i, nqubits - i - 1)

    return tncirc


def init_state_tn(nqubits, init_state_sv):
    dims = tuple(2 * np.ones(nqubits, dtype=int))

    return qtn.tensor_1d.MatrixProductState.from_dense(init_state_sv, dims)


def eval(qasm: str, init_state, backend="numpy", swaps=True):
    """Evaluate QASM with Quimb

    backend (quimb): numpy, cupy, jax. Passed to ``opt_einsum``.

    """
    circuit = QiboCircuit.from_qasm(qasm)
    init_state_mps = init_state_tn(circuit.nqubits, init_state)
    circ_quimb = from_qibo(circuit, swaps=swaps, psi0=init_state_mps)
    interim = circ_quimb.psi.full_simplify(seq="DRC")
    amplitudes = interim.to_dense(backend=backend).flatten()

    return amplitudes

import numpy as np
import quimb.tensor as qtn


def init_state_tn(nqubits, init_state_sv):
    """Create a matrix product state directly from a dense vector."""

    dims = tuple(2 * np.ones(nqubits, dtype=int))

    return qtn.tensor_1d.MatrixProductState.from_dense(init_state_sv, dims)


def dense_vector_tn_qu(qasm: str, initial_state, mps_opts, backend="numpy"):
    """Evaluate QASM with Quimb.

    backend (quimb): numpy, cupy, jax. Passed to ``opt_einsum``.
    """

    if initial_state is not None:
        nqubits = int(np.log2(len(initial_state)))
        initial_state = init_state_tn(nqubits, initial_state)

    if mps_opts:
        circ_quimb = qtn.circuit.CircuitMPS.from_openqasm2_str(
            qasm, psi0=initial_state, gate_opts=mps_opts
        )

    else:
        circ_quimb = qtn.circuit.Circuit.from_openqasm2_str(qasm, psi0=initial_state)

    interim = circ_quimb.psi.full_simplify(seq="DRC")
    amplitudes = interim.to_dense(backend=backend)

    return amplitudes

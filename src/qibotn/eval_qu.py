import numpy as np
import quimb.tensor as qtn


def init_state_tn(nqubits, init_state_sv):
    """Create a matrix product state directly from a dense vector.

    Args:
        nqubits (int): Total number of qubits in the circuit.
        init_state_sv (list): Initial state in the dense vector form.

    Returns:
        list: Matrix product state representation of the dense vector.
    """

    dims = tuple(2 * np.ones(nqubits, dtype=int))
    return qtn.tensor_1d.MatrixProductState.from_dense(init_state_sv, dims)


def dense_vector_tn_qu(qasm: str, initial_state, mps_opts, backend="numpy"):
    """Evaluate circuit in QASM format with Quimb.

    Args:
        qasm (str): QASM program.
        initial_state (list): Initial state in the dense vector form. If ``None`` the default ``|00...0>`` state is used.
        mps_opts (dict): Parameters to tune the gate_opts for mps settings in ``class quimb.tensor.circuit.CircuitMPS``.
        backend (str):  Backend to perform the contraction with, e.g. ``numpy``, ``cupy``, ``jax``. Passed to ``opt_einsum``.

    Returns:
        list: Amplitudes of final state after the simulation of the circuit.
    """

    if initial_state is not None:
        nqubits = int(np.log2(len(initial_state)))
        initial_state = init_state_tn(nqubits, initial_state)

    circ_cls = qtn.circuit.CircuitMPS if mps_opts else qtn.circuit.Circuit
    circ_quimb = circ_cls.from_openqasm2_str(
        qasm, psi0=initial_state, gate_opts=mps_opts
    )

    interim = circ_quimb.psi.full_simplify(seq="DRC")
    amplitudes = interim.to_dense(backend=backend)

    return amplitudes


def tebd_tn_qu(circuit, tebd_opts, mps_opts):
    """Circuit based TEBD which returns the final evolved state as a dense
    vector."""

    init_state = tebd_opts["initial_state"]
    dt = tebd_opts["dt"]
    T = tebd_opts["tot_time"]
    nqubits = circuit.nqubits

    initial_state = qtn.MPS_computational_state(init_state)

    import qiskit.qasm2 as qasm
    from qiskit import QuantumCircuit, transpile

    circ = QuantumCircuit(nqubits)
    unitary = circuit.unitary()

    qubit_arr = []
    for i in range(0, nqubits):
        qubit_arr.append(i)
    circ.unitary(unitary, qubit_arr)

    transpiled_circ = transpile(circ, basis_gates=["rx", "ry", "rz", "cx"])

    qasm_str = qasm.dumps(transpiled_circ)

    circ_cls = qtn.circuit.CircuitMPS
    circ_quimb = circ_cls.from_openqasm2_str(
        qasm_str, psi0=initial_state, gate_opts=mps_opts
    )

    psi0 = initial_state
    for i in np.arange(0, T, dt):

        circ_quimb.psi0 = psi0
        interim = circ_quimb.psi.full_simplify(seq="DRC")
        psi0 = interim  # - should this be done to update the initial state to the next evolving state

    amplitudes = psi0.to_dense()
    return amplitudes


def tebd_tn_qu_2(circuit, tebd_opts):
    """Circuit based TEBD which returns the final evolved state as a dense
    vector."""

    dt = tebd_opts["dt"]
    tot_time = tebd_opts["tot_time"]
    init_state = tebd_opts["initial_state"]
    nqubits = circuit.nqubits

    initial_state = qtn.MPS_computational_state(init_state)

    i = -1
    uni = circuit.unitary()
    h = np.divide((np.log(uni)), -1 * i * dt)

    from qibo import hamiltonians

    ham = hamiltonians.Hamiltonian(nqubits, h)
    ham_quimb = ham.matrix
    H = qtn.LocalHam1D(2, H2=ham_quimb)

    tebd = qtn.TEBD(initial_state, H)

    ts = np.arange(0, 1, dt)
    states = {}
    for t in tebd.at_times(ts, tol=1e-3):
        states.update({None: t.to_dense()})

    state = np.array(list(states.values()))[-1]
    return state

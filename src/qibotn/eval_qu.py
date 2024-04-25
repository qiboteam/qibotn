import numpy as np
import quimb.tensor as qtn
from qibo.config import raise_error


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

def init_state_tn_tebd(initial_state):

    initial_state = qtn.MPS_computational_state(initial_state)
    return initial_state

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

def tebd_tn_qu(tebd_opts, initial_state, nqubits):

    print("Entered TEBD function")
    from qibo.hamiltonians import SymbolicHamiltonian

    init_state = init_state_tn_tebd(initial_state)
    from qibo import hamiltonians

    hamiltonian = tebd_opts["hamiltonian"]
    dt = tebd_opts["dt"]
    initial_state = tebd_opts["initial_state"]
    tot_time = tebd_opts["tot_time"]

    if hamiltonian == "TFIM":
        ham = hamiltonians.TFIM(nqubits=nqubits, dense=False)
    elif hamiltonian == "NIX":
        ham = hamiltonians.X(nqubits=nqubits, dense=False)
    elif hamiltonian == "NIY":
        ham = hamiltonians.Y(nqubits=nqubits, dense=False)
    elif hamiltonian == "NIZ":
        ham = hamiltonians.Z(nqubits=nqubits, dense=False)
    elif hamiltonian == "XXZ":
        ham = hamiltonians.XXZ(nqubits=nqubits, dense=False)
    elif hamiltonian == "MC":
        ham = hamiltonians.MaxCut(nqubits=nqubits, dense=False)
    else:
        raise_error(
            NotImplementedError, "QiboTN does not support custom hamiltonians"
            )

    terms_dict = {}
    i=0
    list_of_terms = ham.terms
    for t in list_of_terms:
        terms_dict.update({None: t.matrix})
        i=i+1
    
    H = qtn.LocalHam1D(nqubits, H2=terms_dict)

    tebd = qtn.TEBD(init_state, H)
    ts = np.arange(0, tot_time, dt)

    file_path = "tebd_qibotn.txt"
    with open(file_path, 'w') as file:
        for t in tebd.at_times(ts, tol=1e-3):
            file.write(+str(t.to_dense()))
    
    with open(file_path, 'r') as file:
        content = file.read()

    state_str = (content.rsplit(']]',2)[-2])+"]]"

    state = np.array(eval(state_str))
    return state

import numpy as np
import quimb.tensor as qtn
from qibo.config import raise_error


def init_state_tn_tebd(initial_state):
    """Creates a inital MPS from a binary string."""

    initial_state = qtn.MPS_computational_state(initial_state)
    return initial_state


def tebd_quimb(circuit, tebd_opts):

    '''Symbolic Hamiltonian based TEBD which returns the final evolved state as a dense vector'''

    from qibo.hamiltonians import SymbolicHamiltonian

    hamiltonian = tebd_opts["hamiltonian"]
    dt = tebd_opts["dt"]
    initial_state = tebd_opts["initial_state"]
    tot_time = tebd_opts["tot_time"]
    nqubits = circuit.nqubits

    init_state = init_state_tn_tebd(initial_state)
    from qibo import hamiltonians

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
        raise_error(NotImplementedError, "QiboTN does not support custom hamiltonians")

    terms_dict = {}
    i = 0
    list_of_terms = ham.terms
    for t in list_of_terms:
        terms_dict.update({None: t.matrix})
        i = i + 1

    H = qtn.LocalHam1D(nqubits, H2=terms_dict)

    tebd = qtn.TEBD(init_state, H)
    ts = np.arange(0, tot_time, dt)

    states = {}
    for t in tebd.at_times(ts, tol=1e-3):
        states.update({None: t.to_dense()})

    state = np.array(list(states.values()))[-1]

    return state

def tebd_quimb_execute_circuit(circuit, tebd_opts, mps_opts):

    initial_state = tebd_opts["initial_state"]
    init_state = init_state_tn_tebd(initial_state)

    circ_cls = qtn.circuit.CircuitMPS if mps_opts else qtn.circuit.Circuit
    circ_quimb = circ_cls.from_openqasm2_str(
        circuit.to_qasm(), psi0=initial_state, gate_opts=mps_opts
    )

    #return state
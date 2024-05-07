import copy
import os

import config
import numpy as np
import pytest
import qibo
from qibo import hamiltonians


def create_init_state(nqubits):
    init_state = np.ones(nqubits)
    return init_state

def qibo_trotter(nqubits, init_state, dt):
    ham = hamiltonians.XXZ(nqubits=nqubits)
    circ_qibo = ham.circuit(dt=dt)
    state_vec = circ_qibo(init_state).state(numpy=True)
    return circ_qibo, state_vec

@pytest.mark.parametrize(
    "nqubits, tolerance, is_tebd",
    [(4, 1e-6, True), (5, 1e-6, False), (6, 1e-3, True), (10, 1e-3, False)],
)
def test_eval(nqubits: int, tolerance: float, is_tebd: bool):
    """Evaluate circuit with Quimb backend.

    Args:
        nqubits (int): Total number of qubits in the system.
        tolerance (float): Maximum limit allowed for difference in results
        is_tebd (bool): True if user selects is TEBD and False if otherwise
    """
    # hack quimb to use the correct number of processes
    # TODO: remove completely, or at least delegate to the backend
    # implementation
    os.environ["QUIMB_NUM_PROCS"] = str(os.cpu_count())
    import qibotn.eval_qu

    init_state = create_init_state(nqubits=nqubits)
    init_state_tn = copy.deepcopy(init_state)

    # Test qibo
    qibo.set_backend(backend=config.qibo.backend, platform=config.qibo.platform)

    qibo_circ, result_sv = qibo_trotter(nqubits, init_state, dt=1e-4)

    # Test quimb
    if is_tebd:
        gate_opt = {}
        gate_opt["dt"] = 1e-4
        gate_opt["hamiltonian"] = "XXZ"
        gate_opt["initial_state"] = "11111"
        gate_opt["tot_time"] = 1
    else:
        gate_opt = None
    result_tn = qibotn.tebd.tebd_quimb(
        qibo_circ, init_state_tn, gate_opt, backend=config.quimb.backend
    ).flatten()
    
    assert np.allclose(
        result_sv, result_tn, atol=tolerance
    ), "Resulting dense vectors do not match"




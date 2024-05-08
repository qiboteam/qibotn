import copy
import os

import config
import numpy as np
import pytest
import qibo
from qibo import Circuit, gates


def create_init_state(nqubits):
    init_state = np.ones(nqubits)
    return init_state

def qibo_crt(nqubits, init_state, dt):
    from numpy import pi
    circ_qibo = Circuit(3)
    circ_qibo.add(gates.RY(0, theta=pi/2))
    circ_qibo.add(gates.RY(1, theta=pi/4))
    circ_qibo.add(gates.CNOT(0,1))
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
    import eval_qu as ev

    qibo_circ, result_sv = qibo_crt(nqubits, init_state, dt=1e-4)

    # Test quimb
    if is_tebd:
        gate_opt = {}
        gate_opt["dt"] = 1e-4
        gate_opt["initial_state"] = "101"
        gate_opt["tot_time"] = 1
    else:
        gate_opt = None
    result_tn = ev.tebd_tn_qu(
        qibo_circ, gate_opt
    ).flatten()
    
    assert np.allclose(
        result_sv, result_tn, atol=tolerance
    ), "Resulting dense vectors do not match"

print(test_eval(nqubits=3, tolerance=1e-6, is_tebd=True))




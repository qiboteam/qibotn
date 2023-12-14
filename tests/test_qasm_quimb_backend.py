import copy
import os
import config
import numpy as np
import pytest
import qibo
from qibo.models import QFT


def create_init_state(nqubits):
    init_state = np.random.random(2**nqubits) + \
        1j * np.random.random(2**nqubits)
    init_state = init_state / np.sqrt((np.abs(init_state) ** 2).sum())
    return init_state


def qibo_qft(nqubits, init_state, swaps):
    circ_qibo = QFT(nqubits, swaps)
    state_vec = circ_qibo(init_state).state(numpy=True)
    return circ_qibo, state_vec


@pytest.mark.parametrize("nqubits, tolerance, is_mps",
                         [(1, 1e-6, True), (2, 1e-6, False), (5, 1e-3, True), (10, 1e-3, False)])
def test_eval(nqubits: int, tolerance: float, is_mps: bool):
    # hack quimb to use the correct number of processes
    # TODO: remove completely, or at least delegate to the backend
    # implementation
    os.environ["QUIMB_NUM_PROCS"] = str(os.cpu_count())
    import qibotn.quimb

    init_state = create_init_state(nqubits=nqubits)
    init_state_tn = copy.deepcopy(init_state)

    # Test qibo
    qibo.set_backend(backend=config.qibo.backend,
                     platform=config.qibo.platform)
    #qibo_time, (qibo_circ, result_sv) = time(
        #lambda: qibo_qft(nqubits, init_state, swaps=True)
    #)
    qibo_circ, result_sv= qibo_qft(nqubits, init_state, swaps=True)
    

    # Convert to qasm for other backends
    qasm_circ = qibo_circ.to_qasm()

    # Test quimb
    result_tn = qibotn.quimb.eval(
            qasm_circ, init_state_tn, is_mps, backend=config.quimb.backend
        )
   

    assert np.allclose(result_sv, result_tn,
                       atol=tolerance), "Resulting dense vectors do not match"

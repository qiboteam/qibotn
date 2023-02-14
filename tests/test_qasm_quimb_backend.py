import os

import pytest
import qibo
from qibo.models import QFT
import numpy as np
import copy
from timeit import default_timer as timer

import config


def init_state_sv(nqubits):
    init_state = np.random.random(2**nqubits) + 1j * np.random.random(2**nqubits)
    init_state = init_state / np.sqrt((np.abs(init_state) ** 2).sum())
    # An unmodified init_state has to be converted to tn format
    init_state_for_tn = copy.deepcopy(init_state)
    return init_state, init_state_for_tn


def qibo_qft(nqubits, init_state, swaps):
    circ_qibo = QFT(nqubits, swaps)
    state_vec = np.array(circ_qibo(init_state))
    return circ_qibo, state_vec


@pytest.mark.parametrize("nqubits", [1, 2, 5, 10])
def test_eval(nqubits: int):
    os.environ["QUIMB_NUM_PROCS"] = str(os.cpu_count())
    from qibotn import qasm_quimb

    init_state_qibo, init_state_for_tn = init_state_sv(nqubits=nqubits)

    # Test qibo
    qibo.set_backend(backend=config.qibo['backend'], \
        platform=config.qibo['platform'])
    start_time = timer()
    qibo_circ, result_sv = qibo_qft(nqubits, init_state=init_state_qibo, \
        swaps=config.qibo['swaps'])
    end_time = timer()
    qibo_time = end_time - start_time

    # Convert to qasm for other backends
    qasm_circ = qibo_circ.to_qasm()

    # Test quimb
    start_time = timer()
    result_tn = qasm_quimb.eval_QI_qft(nqubits=nqubits, qasm_circ=qasm_circ, \
        init_state=init_state_for_tn, backend=config.quimb['backend'], \
            swaps=config.quimb['swaps'])
    end_time = timer()
    quimb_time = end_time - start_time

    assert np.allclose(result_sv, result_tn), \
        "Resulting dense vectors do not match"

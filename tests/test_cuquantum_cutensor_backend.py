import copy
import os
from timeit import default_timer as timer

import config
import numpy as np
import pytest
import qibo
from qibo.models import QFT


def qibo_qft(nqubits, swaps):
    circ_qibo = QFT(nqubits, swaps)
    state_vec = np.array(circ_qibo())
    return circ_qibo, state_vec


def time(func):
    start = timer()
    res = func()
    end = timer()
    time = end - start
    return time, res


@pytest.mark.parametrize("nqubits", [1, 2, 5, 10])
def test_eval(nqubits: int):
    import qibotn.cutn

    # Test qibo
    qibo.set_backend(backend=config.qibo.backend, platform=config.qibo.platform)
    qibo_time, (qibo_circ, result_sv) = time(lambda: qibo_qft(nqubits, swaps=True))

    # Test Cuquantum
    cutn_time, result_tn = time(lambda: qibotn.cutn.eval(qibo_circ))

    assert 1e-2 * qibo_time < cutn_time < 1e2 * qibo_time
    assert np.allclose(result_sv, result_tn), "Resulting dense vectors do not match"

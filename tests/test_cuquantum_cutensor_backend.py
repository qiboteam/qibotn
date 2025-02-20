import math
from timeit import default_timer as timer

import cupy as cp
import numpy as np
import pytest
import qibo
from qibo import Circuit, construct_backend, gates, hamiltonians
from qibo.models import QFT
from qibo.symbols import X, Z


def qibo_qft(nqubits, swaps):
    circ_qibo = QFT(nqubits, swaps)
    state_vec = circ_qibo().state(numpy=True)
    return circ_qibo, state_vec


def time(func):
    start = timer()
    res = func()
    end = timer()
    time = end - start
    return time, res


def build_observable(nqubits):
    """Helper function to construct a target observable."""
    hamiltonian_form = 0
    for i in range(nqubits):
        hamiltonian_form += 0.5 * X(i % nqubits) * Z((i + 1) % nqubits)

    hamiltonian = hamiltonians.SymbolicHamiltonian(form=hamiltonian_form)
    return hamiltonian, hamiltonian_form


@pytest.mark.gpu
@pytest.mark.parametrize("nqubits", [1, 2, 5, 10])
def test_eval(nqubits: int, dtype="complex128"):
    """Evaluate QASM with cuQuantum.

    Args:
        nqubits (int): Total number of qubits in the system.
        dtype (str): The data type for precision, 'complex64' for single,
            'complex128' for double.
    """
    import qibotn.eval

    # Test qibo
    # qibo.set_backend(backend=config.qibo.backend, platform=config.qibo.platform)
    qibo.set_backend(backend="numpy")
    qibo_time, (qibo_circ, result_sv) = time(lambda: qibo_qft(nqubits, swaps=True))
    result_sv_cp = cp.asarray(result_sv)

    # Test Cuquantum
    cutn_time, result_tn = time(
        lambda: qibotn.eval.dense_vector_tn(qibo_circ, dtype).flatten()
    )

    print(f"State vector difference: {abs(result_tn - result_sv_cp).max():0.3e}")

    assert cp.allclose(result_sv_cp, result_tn), "Resulting dense vectors do not match"


@pytest.mark.gpu
@pytest.mark.parametrize("nqubits", [2, 5, 10])
def test_mps(nqubits: int, dtype="complex128"):
    """Evaluate MPS with cuQuantum.

    Args:
        nqubits (int): Total number of qubits in the system.
        dtype (str): The data type for precision, 'complex64' for single,
            'complex128' for double.
    """
    import qibotn.eval

    # Test qibo
    qibo.set_backend(backend="numpy")

    qibo_time, (circ_qibo, result_sv) = time(lambda: qibo_qft(nqubits, swaps=True))

    result_sv_cp = cp.asarray(result_sv)

    # Test of MPS
    gate_algo = {
        "qr_method": False,
        "svd_method": {
            "partition": "UV",
            "abs_cutoff": 1e-12,
        },
    }

    cutn_time, result_tn = time(
        lambda: qibotn.eval.dense_vector_mps(circ_qibo, gate_algo, dtype).flatten()
    )

    print(f"State vector difference: {abs(result_tn - result_sv_cp).max():0.3e}")

    assert cp.allclose(result_tn, result_sv_cp), "Resulting dense vectors do not match"


@pytest.mark.gpu
@pytest.mark.parametrize("nqubits", [2, 5, 10])
def test_expectation(nqubits: int, dtype="complex128"):
    import qibotn.eval

    circ_qibo, state_vec_qibo = qibo_qft(nqubits, swaps=True)
    ham, ham_form = build_observable(nqubits)

    numpy_backend = construct_backend("numpy")
    exact_expval = numpy_backend.calculate_expectation_state(
        hamiltonian=ham,
        state=state_vec_qibo,
        normalize=False,
    )

    tn_expval = qibotn.eval.expectation_tn(circ_qibo, dtype, ham).flatten()

    assert math.isclose(exact_expval.item(), tn_expval.real.get().item(), abs_tol=1e-7)

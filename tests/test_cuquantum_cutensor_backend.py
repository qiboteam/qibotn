import math

import cupy as cp
import pytest
import qibo
from qibo import construct_backend, hamiltonians
from qibo.models import QFT
from qibo.symbols import X, Z


def qibo_qft(nqubits, swaps):
    circ_qibo = QFT(nqubits, swaps)
    state_vec = circ_qibo().state(numpy=True)
    return circ_qibo, state_vec


def build_observable(nqubits):
    """Helper function to construct a target observable."""
    hamiltonian_form = 0
    for i in range(nqubits):
        hamiltonian_form += 0.5 * X(i % nqubits) * Z((i + 1) % nqubits)

    hamiltonian = hamiltonians.SymbolicHamiltonian(form=hamiltonian_form)
    return hamiltonian, hamiltonian_form


def build_observable_dict(nqubits):
    """Construct a target observable as a dictionary representation.

    Returns a dictionary suitable for `create_hamiltonian_from_dict`.
    """
    terms = []

    for i in range(nqubits):
        term = {
            "coefficient": 0.5,
            "operators": [("X", i % nqubits), ("Z", (i + 1) % nqubits)],
        }
        terms.append(term)

    return {"terms": terms}


@pytest.mark.gpu
@pytest.mark.parametrize("nqubits", [1, 2, 5, 10])
def test_eval(nqubits: int, dtype="complex128"):
    """
    Args:
        nqubits (int): Total number of qubits in the system.
        dtype (str): The data type for precision, 'complex64' for single,
            'complex128' for double.
    """
    # Test qibo
    qibo.set_backend(backend="numpy")
    qibo_circ, result_sv = qibo_qft(nqubits, swaps=True)
    result_sv_cp = cp.asarray(result_sv)

    # Test cutensornet
    backend = construct_backend(backend="qibotn", platform="cutensornet")
    # Test with no settings specified. Default is dense vector calculation without MPI or NCCL.
    result_tn = backend.execute_circuit(circuit=qibo_circ)
    print(
        f"State vector difference: {abs(result_tn.statevector.flatten() - result_sv_cp).max():0.3e}"
    )
    assert cp.allclose(
        result_sv_cp, result_tn.statevector.flatten()
    ), "Resulting dense vectors do not match"

    # Test with explicit settings specified.
    computation_settings = {
        "MPI_enabled": False,
        "MPS_enabled": False,
        "NCCL_enabled": False,
        "expectation_enabled": False,
    }
    backend.configure_tn_simulation(computation_settings)
    result_tn = backend.execute_circuit(circuit=qibo_circ)
    print(
        f"State vector difference: {abs(result_tn.statevector.flatten() - result_sv_cp).max():0.3e}"
    )
    assert cp.allclose(
        result_sv_cp, result_tn.statevector.flatten()
    ), "Resulting dense vectors do not match"


@pytest.mark.gpu
@pytest.mark.parametrize("nqubits", [2, 5, 10])
def test_mps(nqubits: int, dtype="complex128"):
    """Evaluate MPS with cuQuantum.

    Args:
        nqubits (int): Total number of qubits in the system.
        dtype (str): The data type for precision, 'complex64' for single,
            'complex128' for double.
    """

    # Test qibo
    qibo.set_backend(backend="numpy")
    qibo_circ, result_sv = qibo_qft(nqubits, swaps=True)
    result_sv_cp = cp.asarray(result_sv)

    # Test cutensornet
    backend = construct_backend(backend="qibotn", platform="cutensornet")
    # Test with simple MPS settings specified using bool. Uses the default MPS parameters.
    computation_settings_1 = {
        "MPI_enabled": False,
        "MPS_enabled": True,
        "NCCL_enabled": False,
        "expectation_enabled": False,
    }
    backend.configure_tn_simulation(computation_settings_1)
    result_tn = backend.execute_circuit(circuit=qibo_circ)
    print(
        f"State vector difference: {abs(result_tn.statevector.flatten() - result_sv_cp).max():0.3e}"
    )
    assert cp.allclose(
        result_tn.statevector.flatten(), result_sv_cp
    ), "Resulting dense vectors do not match"

    # Test with explicit MPS computation settings specified using Dict. Users able to specify parameters like qr_method etc.
    computation_settings_2 = {
        "MPI_enabled": False,
        "MPS_enabled": {
            "qr_method": False,
            "svd_method": {
                "partition": "UV",
                "abs_cutoff": 1e-12,
            },
        },
        "NCCL_enabled": False,
        "expectation_enabled": False,
    }
    backend.configure_tn_simulation(computation_settings_2)
    result_tn = backend.execute_circuit(circuit=qibo_circ)
    print(
        f"State vector difference: {abs(result_tn.statevector.flatten() - result_sv_cp).max():0.3e}"
    )
    assert cp.allclose(
        result_tn.statevector.flatten(), result_sv_cp
    ), "Resulting dense vectors do not match"


@pytest.mark.parametrize("nqubits", [2, 5, 10])
def test_expectation(nqubits: int, dtype="complex128"):

    # Test qibo
    qibo_circ, state_vec_qibo = qibo_qft(nqubits, swaps=True)
    ham, ham_form = build_observable(nqubits)
    numpy_backend = construct_backend("numpy")
    exact_expval = numpy_backend.calculate_expectation_state(
        hamiltonian=ham,
        state=state_vec_qibo,
        normalize=False,
    )

    # Test cutensornet
    backend = construct_backend(backend="qibotn", platform="cutensornet")

    # Test with simple settings using bool. Uses default Hamilitonian for expectation calculation.
    computation_settings_1 = {
        "MPI_enabled": False,
        "MPS_enabled": False,
        "NCCL_enabled": False,
        "expectation_enabled": True,
    }
    backend.configure_tn_simulation(computation_settings_1)
    result_tn = backend.execute_circuit(circuit=qibo_circ)
    assert math.isclose(exact_expval.item(), result_tn.real.get().item(), abs_tol=1e-7)

    # Test with user defined hamiltonian using "hamiltonians.SymbolicHamiltonian" object.
    computation_settings_2 = {
        "MPI_enabled": False,
        "MPS_enabled": False,
        "NCCL_enabled": False,
        "expectation_enabled": ham,
    }
    backend.configure_tn_simulation(computation_settings_2)
    result_tn = backend.execute_circuit(circuit=qibo_circ)
    assert math.isclose(exact_expval.item(), result_tn.real.get().item(), abs_tol=1e-7)

    # Test with user defined hamiltonian using Dictionary object form of hamiltonian.
    ham_dict = build_observable_dict(nqubits)
    computation_settings_3 = {
        "MPI_enabled": False,
        "MPS_enabled": False,
        "NCCL_enabled": False,
        "expectation_enabled": ham_dict,
    }
    backend.configure_tn_simulation(computation_settings_3)
    result_tn = backend.execute_circuit(circuit=qibo_circ)
    assert math.isclose(exact_expval.item(), result_tn.real.get().item(), abs_tol=1e-7)

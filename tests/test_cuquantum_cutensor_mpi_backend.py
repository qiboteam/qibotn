# mpirun --allow-run-as-root -np 2 python -m pytest --with-mpi test_cuquantum_cutensor_mpi_backend.py

import math

import cupy as cp
import numpy as np
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
@pytest.mark.mpi
@pytest.mark.parametrize("nqubits", [1, 2, 5, 7, 10])
def test_eval_mpi(nqubits: int, dtype="complex128"):
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

    # Test 1: Explicit computation settings specified (same as default).
    computation_settings = {
        "MPI_enabled": True,
        "MPS_enabled": False,
        "NCCL_enabled": False,
        "expectation_enabled": False,
    }
    backend.configure_tn_simulation(computation_settings)
    result_tn = backend.execute_circuit(circuit=qibo_circ)
    result_tn_cp = cp.asarray(result_tn.statevector.flatten())

    print(f"State vector difference: {abs(result_tn_cp - result_sv_cp).max():0.3e}")

    if backend.rank == 0:

        assert cp.allclose(
            result_sv_cp, result_tn_cp
        ), "Resulting dense vectors do not match"
    else:
        assert (
            isinstance(result_tn_cp, cp.ndarray)
            and result_tn_cp.size == 1
            and result_tn_cp.item() == 0
        ), f"Rank {backend.rank}: result_tn_cp should be scalar/array with 0, got {result_tn_cp}"

@pytest.mark.gpu
@pytest.mark.mpi
@pytest.mark.parametrize("nqubits", [1, 2, 5, 7, 10])
def test_expectation_mpi(nqubits: int, dtype="complex128"):

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

    # Test 1: No Hamilitonian computation settings specified. Use default.
    computation_settings_1 = {
        "MPI_enabled": True,
        "MPS_enabled": False,
        "NCCL_enabled": False,
        "expectation_enabled": True,
    }
    backend.configure_tn_simulation(computation_settings_1)
    result_tn = backend.execute_circuit(circuit=qibo_circ)
    if backend.rank == 0:
        # Compare numerical values
        assert math.isclose(
            exact_expval.item(), float(result_tn[0]), abs_tol=1e-7
        ), f"Rank {backend.rank}: mismatch, expected {exact_expval}, got {result_tn}"

    else:
        # Rank > 0: must be hardcoded [0] (int)
        assert (
            isinstance(result_tn, (np.ndarray, cp.ndarray))
            and result_tn.size == 1
            and np.issubdtype(result_tn.dtype, np.integer)
            and result_tn.item() == 0
        ), f"Rank {backend.rank}: expected int array [0], got {result_tn}"

    # Test 2: hamiltonians.SymbolicHamiltonian object in computation settings specified.
    computation_settings_2 = {
        "MPI_enabled": True,
        "MPS_enabled": False,
        "NCCL_enabled": False,
        "expectation_enabled": ham,
    }
    backend.configure_tn_simulation(computation_settings_2)
    result_tn = backend.execute_circuit(circuit=qibo_circ)
    if backend.rank == 0:
        # Compare numerical values
        assert math.isclose(
            exact_expval.item(), float(result_tn[0]), abs_tol=1e-7
        ), f"Rank {backend.rank}: mismatch, expected {exact_expval}, got {result_tn}"

    else:
        # Rank > 0: must be hardcoded [0] (int)
        assert (
            isinstance(result_tn, (np.ndarray, cp.ndarray))
            and result_tn.size == 1
            and np.issubdtype(result_tn.dtype, np.integer)
            and result_tn.item() == 0
        ), f"Rank {backend.rank}: expected int array [0], got {result_tn}"

    # Test 3: Dictionary object form of hamiltonian in computation settings specified.
    ham_dict = build_observable_dict(nqubits)
    computation_settings_3 = {
        "MPI_enabled": True,
        "MPS_enabled": False,
        "NCCL_enabled": False,
        "expectation_enabled": ham_dict,
    }
    backend.configure_tn_simulation(computation_settings_3)
    result_tn = backend.execute_circuit(circuit=qibo_circ)
    if backend.rank == 0:
        # Compare numerical values
        assert math.isclose(
            exact_expval.item(), float(result_tn[0]), abs_tol=1e-7
        ), f"Rank {backend.rank}: mismatch, expected {exact_expval}, got {result_tn}"

    else:
        # Rank > 0: must be hardcoded [0] (int)
        assert (
            isinstance(result_tn, (np.ndarray, cp.ndarray))
            and result_tn.size == 1
            and np.issubdtype(result_tn.dtype, np.integer)
            and result_tn.item() == 0
        ), f"Rank {backend.rank}: expected int array [0], got {result_tn}"

@pytest.mark.gpu
@pytest.mark.mpi
@pytest.mark.parametrize("nqubits", [1, 2, 5, 7, 10])
def test_eval_nccl(nqubits: int, dtype="complex128"):
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

    # Test 1: Explicit computation settings specified (same as default).
    computation_settings = {
        "MPI_enabled": False,
        "MPS_enabled": False,
        "NCCL_enabled": True,
        "expectation_enabled": False,
    }
    backend.configure_tn_simulation(computation_settings)
    result_tn = backend.execute_circuit(circuit=qibo_circ)
    result_tn_cp = cp.asarray(result_tn.statevector.flatten())

    if backend.rank == 0:
        assert cp.allclose(
            result_sv_cp, result_tn_cp
        ), "Resulting dense vectors do not match"
    else:
        assert (
            isinstance(result_tn_cp, cp.ndarray)
            and result_tn_cp.size == 1
            and result_tn_cp.item() == 0
        ), f"Rank {backend.rank}: result_tn_cp should be scalar/array with 0, got {result_tn_cp}"

@pytest.mark.gpu
@pytest.mark.mpi
@pytest.mark.parametrize("nqubits", [1, 2, 5, 7, 10])
def test_expectation_NCCL(nqubits: int, dtype="complex128"):

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

    # Test 1: No Hamilitonian computation settings specified. Use default.
    computation_settings_1 = {
        "MPI_enabled": False,
        "MPS_enabled": False,
        "NCCL_enabled": True,
        "expectation_enabled": True,
    }
    backend.configure_tn_simulation(computation_settings_1)
    result_tn = backend.execute_circuit(circuit=qibo_circ)
    if backend.rank == 0:
        # Compare numerical values
        assert math.isclose(
            exact_expval.item(), float(result_tn[0]), abs_tol=1e-7
        ), f"Rank {backend.rank}: mismatch, expected {exact_expval}, got {result_tn}"

    else:
        # Rank > 0: must be hardcoded [0] (int)
        assert (
            isinstance(result_tn, (np.ndarray, cp.ndarray))
            and result_tn.size == 1
            and np.issubdtype(result_tn.dtype, np.integer)
            and result_tn.item() == 0
        ), f"Rank {backend.rank}: expected int array [0], got {result_tn}"

    # Test 2: hamiltonians.SymbolicHamiltonian object in computation settings specified.
    computation_settings_2 = {
        "MPI_enabled": False,
        "MPS_enabled": False,
        "NCCL_enabled": True,
        "expectation_enabled": ham,
    }
    backend.configure_tn_simulation(computation_settings_2)
    result_tn = backend.execute_circuit(circuit=qibo_circ)
    if backend.rank == 0:
        # Compare numerical values
        assert math.isclose(
            exact_expval.item(), float(result_tn[0]), abs_tol=1e-7
        ), f"Rank {backend.rank}: mismatch, expected {exact_expval}, got {result_tn}"

    else:
        # Rank > 0: must be hardcoded [0] (int)
        assert (
            isinstance(result_tn, (np.ndarray, cp.ndarray))
            and result_tn.size == 1
            and np.issubdtype(result_tn.dtype, np.integer)
            and result_tn.item() == 0
        ), f"Rank {backend.rank}: expected int array [0], got {result_tn}"

    # Test 3: Dictionary object form of hamiltonian in computation settings specified.
    ham_dict = build_observable_dict(nqubits)
    computation_settings_3 = {
        "MPI_enabled": False,
        "MPS_enabled": False,
        "NCCL_enabled": True,
        "expectation_enabled": ham_dict,
    }
    backend.configure_tn_simulation(computation_settings_3)
    result_tn = backend.execute_circuit(circuit=qibo_circ)
    if backend.rank == 0:
        # Compare numerical values
        assert math.isclose(
            exact_expval.item(), float(result_tn[0]), abs_tol=1e-7
        ), f"Rank {backend.rank}: mismatch, expected {exact_expval}, got {result_tn}"

    else:
        # Rank > 0: must be hardcoded [0] (int)
        assert (
            isinstance(result_tn, (np.ndarray, cp.ndarray))
            and result_tn.size == 1
            and np.issubdtype(result_tn.dtype, np.integer)
            and result_tn.item() == 0
        ), f"Rank {backend.rank}: expected int array [0], got {result_tn}"

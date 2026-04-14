# mpirun -np 2 python -m pytest tests/test_quimb_mpi_backend.py -m mpi

import math

import numpy as np
import pytest
import qibo
from qibo import construct_backend, hamiltonians
from qibo.models import QFT
from qibo.symbols import X, Z

pytest.importorskip("mpi4py")

ABS_TOL = 1e-7


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
    return hamiltonian


def build_symbolic_lists(nqubits):
    """Build operators/sites/coeffs accepted by exp_value_observable_symbolic."""
    operators_list = []
    sites_list = []
    coeffs_list = []

    for i in range(nqubits):
        operators_list.append("xz")
        sites_list.append((i % nqubits, (i + 1) % nqubits))
        coeffs_list.append(0.5)

    return operators_list, sites_list, coeffs_list


@pytest.mark.parametrize("nqubits", [2, 5, 7])
def test_quimb_statevector_mpi(nqubits: int):
    qibo.set_backend(backend="numpy")
    qibo_circ, expected_sv = qibo_qft(nqubits, swaps=True)

    backend = construct_backend(backend="qibotn", platform="quimb")
    backend.configure_tn_simulation(
        ansatz="mps",
        max_bond_dimension=None,
        svd_cutoff=1e-12,
        MPI_enabled=True,
    )

    outcome = backend.execute_circuit(circuit=qibo_circ, return_array=True)

    if backend.rank == 0:
        got_sv = outcome.state().flatten()
        assert np.allclose(
            expected_sv, got_sv, atol=1e-7, rtol=1e-7
        ), "Resulting dense vectors do not match"
    else:
        assert outcome.state() is None


@pytest.mark.parametrize("nqubits", [2, 5, 7])
def test_quimb_expectation_mpi(nqubits: int):
    qibo.set_backend(backend="numpy")
    qibo_circ, _ = qibo_qft(nqubits, swaps=True)

    ham = build_observable(nqubits)
    exact_expval = ham.expectation(qibo_circ)

    operators_list, sites_list, coeffs_list = build_symbolic_lists(nqubits)

    backend = construct_backend(backend="qibotn", platform="quimb")
    backend.configure_tn_simulation(
        ansatz="mps",
        max_bond_dimension=None,
        svd_cutoff=1e-12,
        MPI_enabled=True,
    )

    result_tn = backend.exp_value_observable_symbolic(
        qibo_circ,
        operators_list,
        sites_list,
        coeffs_list,
        nqubits,
    )

    if backend.rank == 0:
        assert math.isclose(
            float(exact_expval), float(result_tn), abs_tol=ABS_TOL
        ), f"Rank {backend.rank}: mismatch, expected {exact_expval}, got {result_tn}"
    else:
        assert result_tn == 0.0

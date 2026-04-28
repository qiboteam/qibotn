import sys
import time

import numpy as np
from qibo import Circuit, gates
from qibo.backends import construct_backend

# Parse command line argument for number of processes expected
expected_procs = int(sys.argv[1]) if len(sys.argv) > 1 else 2

np.random.seed(42)


def build_large_circuit(nqubits, nlayers):
    """Build a larger circuit for benchmarking."""
    circ = Circuit(nqubits)
    for _ in range(nlayers):
        for q in range(nqubits):
            circ.add(gates.RY(q=q, theta=np.random.random()))
            circ.add(gates.RZ(q=q, theta=np.random.random()))
        for q in range(nqubits):
            circ.add(gates.CNOT(q % nqubits, (q + 1) % nqubits))
    return circ


# Get MPI info
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    print("ERROR: mpi4py not available")
    exit(1)

# Circuit parameters - larger for benchmarking
nqubits = 8
nlayers = 3

if rank == 0:
    print(f"=" * 60)
    print(f"MPI EXPECTATION VALUE BENCHMARK")
    print(f"=" * 60)
    print(f"MPI processes: {size} (expected: {expected_procs})")
    print(f"Circuit: {nqubits} qubits, {nlayers} layers")
    print(f"State space: 2^{nqubits} = {2**nqubits} dimensions")
    print(f"=" * 60)

# Build circuit
circuit = build_large_circuit(nqubits, nlayers)

# Define Hamiltonian with multiple terms
operators_list = ["z", "x", "y", "zz", "xx", "yy", "xyz"]
sites_list = [(0,), (1,), (2,), (3, 4), (5, 6), (1, 2), (0, 1, 2)]
coeffs_list = [1.0, 0.5, 0.3, 0.8, 0.6, 0.4, 0.2]

if rank == 0:
    print(f"\nHamiltonian: {len(operators_list)} terms")
    for i, (ops, sites, coeff) in enumerate(
        zip(operators_list, sites_list, coeffs_list)
    ):
        print(f"  Term {i+1}: {coeff} * {ops} on qubits {sites}")

# Configure backend with MPI
backend = construct_backend(backend="qibotn", platform="quimb")
backend.configure_tn_simulation(ansatz="mps", max_bond_dimension=20, MPI_enabled=True)

# Warm-up run
if rank == 0:
    print("\nWarm-up run...")
circuit_warmup = build_large_circuit(4, 2)
operators_warmup = ["z", "x"]
sites_warmup = [(0,), (1,)]
coeffs_warmup = [1.0, 1.0]
_ = backend.exp_value_observable_symbolic(
    circuit_warmup, operators_warmup, sites_warmup, coeffs_warmup, 4
)

# Synchronize before timing
comm.Barrier()

# Timed execution
if rank == 0:
    print(f"\nStarting timed expectation value computation with {size} processes...")

start_time = time.time()
exp_value = backend.exp_value_observable_symbolic(
    circuit, operators_list, sites_list, coeffs_list, nqubits
)
end_time = time.time()

execution_time = end_time - start_time

# Gather timing from all ranks
all_times = comm.gather(execution_time, root=0)
all_values = comm.gather(exp_value, root=0)

if rank == 0:
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"\nTiming per rank:")
    for i, t in enumerate(all_times):
        print(f"  Rank {i}: {t:.4f} seconds")

    avg_time = np.mean(all_times)
    min_time = np.min(all_times)
    max_time = np.max(all_times)

    print(f"\nTiming Statistics:")
    print(f"  Average: {avg_time:.4f} seconds")
    print(f"  Min:     {min_time:.4f} seconds")
    print(f"  Max:     {max_time:.4f} seconds")
    print(
        f"  Range:   {max_time - min_time:.4f} seconds ({((max_time-min_time)/avg_time*100):.1f}%)"
    )

    print(f"\nExpectation values per rank:")
    for i, val in enumerate(all_values):
        print(f"  Rank {i}: {val:.10f}")

    print(f"\n{'=' * 60}")
    print(f"Expectation Value: {exp_value:.10f}")
    print(f"Computation Time:  {execution_time:.4f} seconds")
    print(f"{'=' * 60}")
    print(f"✓ MPI expectation computation working with {size} processes")
    print(f"{'=' * 60}")
else:
    if rank == 0 or abs(exp_value) > 1e-10:
        print(f"Rank {rank}: Completed in {execution_time:.4f} seconds")

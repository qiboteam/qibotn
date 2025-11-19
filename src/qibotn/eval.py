import cupy as cp
# import cuquantum.tensornet as cutn
import cuquantum.bindings.cutensornet as cutn
from cupy.cuda import nccl
from cupy.cuda.runtime import getDeviceCount
from cuquantum.tensornet import Network, contract
from mpi4py import MPI
from qibo import hamiltonians
from qibo.symbols import I, X, Y, Z

from qibotn.circuit_convertor import QiboCircuitToEinsum
from qibotn.circuit_to_mps import QiboCircuitToMPS
from qibotn.mps_contraction_helper import MPSContractionHelper


def check_observable(observable, circuit_nqubit):
    """Checks the type of observable and returns the appropriate Hamiltonian."""
    if observable is None:
        return build_observable(circuit_nqubit)
    elif isinstance(observable, dict):
        return create_hamiltonian_from_dict(observable, circuit_nqubit)
    elif isinstance(observable, hamiltonians.SymbolicHamiltonian):
        # TODO: check if the observable is compatible with the circuit
        return observable
    else:
        raise TypeError("Invalid observable type.")


def build_observable(circuit_nqubit):
    """Helper function to construct a target observable."""
    hamiltonian_form = 0
    for i in range(circuit_nqubit):
        hamiltonian_form += 0.5 * X(i % circuit_nqubit) * Z((i + 1) % circuit_nqubit)

    hamiltonian = hamiltonians.SymbolicHamiltonian(form=hamiltonian_form)
    return hamiltonian


def create_hamiltonian_from_dict(data, circuit_nqubit):
    """Create a Qibo SymbolicHamiltonian from a dictionary representation.

    Ensures that each Hamiltonian term explicitly acts on all circuit qubits
    by adding identity (`I`) gates where needed.

    Args:
        data (dict): Dictionary containing Hamiltonian terms.
        circuit_nqubit (int): Total number of qubits in the quantum circuit.

    Returns:
        hamiltonians.SymbolicHamiltonian: The constructed Hamiltonian.
    """
    PAULI_GATES = {"X": X, "Y": Y, "Z": Z}

    terms = []

    for term in data["terms"]:
        coeff = term["coefficient"]
        operators = term["operators"]  # List of tuples like [("Z", 0), ("X", 1)]

        # Convert the operator list into a dictionary {qubit_index: gate}
        operator_dict = {q: PAULI_GATES[g] for g, q in operators}

        # Build the full term ensuring all qubits are covered
        full_term_expr = [
            operator_dict[q](q) if q in operator_dict else I(q)
            for q in range(circuit_nqubit)
        ]

        # Multiply all operators together to form a single term
        term_expr = full_term_expr[0]
        for op in full_term_expr[1:]:
            term_expr *= op

        # Scale by the coefficient
        final_term = coeff * term_expr
        terms.append(final_term)

    if not terms:
        raise ValueError("No valid Hamiltonian terms were added.")

    # Combine all terms
    hamiltonian_form = sum(terms)

    return hamiltonians.SymbolicHamiltonian(hamiltonian_form)


def get_ham_gates(pauli_map, dtype="complex128", backend=cp):
    """Populate the gates for all pauli operators.

    Parameters:
        pauli_map: A dictionary mapping qubits to pauli operators.
        dtype: Data type for the tensor operands.
        backend: The package the tensor operands belong to.

    Returns:
        A sequence of pauli gates.
    """
    asarray = backend.asarray
    pauli_i = asarray([[1, 0], [0, 1]], dtype=dtype)
    pauli_x = asarray([[0, 1], [1, 0]], dtype=dtype)
    pauli_y = asarray([[0, -1j], [1j, 0]], dtype=dtype)
    pauli_z = asarray([[1, 0], [0, -1]], dtype=dtype)

    operand_map = {"I": pauli_i, "X": pauli_x, "Y": pauli_y, "Z": pauli_z}
    gates = []
    for qubit, pauli_char, coeff in pauli_map:
        operand = operand_map.get(pauli_char)
        if operand is None:
            raise ValueError("pauli string character must be one of I/X/Y/Z")
        operand = coeff * operand
        gates.append((operand, (qubit,)))
    return gates


def extract_gates_and_qubits(hamiltonian):
    """
    Extracts the gates and their corresponding qubits from a Qibo Hamiltonian.

    Parameters:
        hamiltonian (qibo.hamiltonians.Hamiltonian or qibo.hamiltonians.SymbolicHamiltonian):
            A Qibo Hamiltonian object.

    Returns:
        list of tuples: [(coefficient, [(gate, qubit), ...]), ...]
            - coefficient: The prefactor of the term.
            - list of (gate, qubit): Each term's gates and the qubits they act on.
    """
    extracted_terms = []

    if isinstance(hamiltonian, hamiltonians.SymbolicHamiltonian):
        for term in hamiltonian.terms:
            coeff = term.coefficient  # Extract coefficient
            gate_qubit_list = []

            # Extract gate and qubit information
            for factor in term.factors:
                gate_name = str(factor)[
                    0
                ]  # Extract the gate type (X, Y, Z) from 'X0', 'Z1'
                qubit = int(str(factor)[1:])  # Extract the qubit index
                gate_qubit_list.append((qubit, gate_name, coeff))
                coeff = 1.0

            extracted_terms.append(gate_qubit_list)

    else:
        raise ValueError(
            "Unsupported Hamiltonian type. Must be SymbolicHamiltonian or Hamiltonian."
        )

    return extracted_terms


def initialize_mpi():
    """Initialize MPI communication and device selection."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    device_id = rank % getDeviceCount()
    cp.cuda.Device(device_id).use()
    return comm, rank, size, device_id


def initialize_nccl(comm_mpi, rank, size):
    """Initialize NCCL communication."""
    nccl_id = nccl.get_unique_id() if rank == 0 else None
    nccl_id = comm_mpi.bcast(nccl_id, root=0)
    return nccl.NcclCommunicator(size, nccl_id, rank)


def get_operands(qibo_circ, datatype, rank, comm):
    """Perform circuit conversion and broadcast operands."""
    if rank == 0:
        myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
        operands = myconvertor.state_vector_operands()
    else:
        operands = None
    return comm.bcast(operands, root=0)


def compute_optimal_path(network, n_samples, size, comm):
    """Compute contraction path and broadcast optimal selection."""
    path, info = network.contract_path(
        optimize={
            "samples": n_samples,
            "slicing": {
                "min_slices": max(32, size),
                "memory_model": cutn.MemoryModel.CUTENSOR,
            },
        }
    )
    opt_cost, sender = comm.allreduce(
        sendobj=(info.opt_cost, comm.Get_rank()), op=MPI.MINLOC
    )
    return comm.bcast(info, sender)


def compute_slices(info, rank, size):
    """Determine the slice range each process should compute."""
    num_slices = info.num_slices
    chunk, extra = num_slices // size, num_slices % size
    slice_begin = rank * chunk + min(rank, extra)
    slice_end = (
        num_slices if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
    )
    return range(slice_begin, slice_end)


def reduce_result(result, comm, method="MPI", root=0):
    """Reduce results across processes."""
    if method == "MPI":
        return comm.reduce(sendobj=result, op=MPI.SUM, root=root)

    elif method == "NCCL":
        stream_ptr = cp.cuda.get_current_stream().ptr
        if result.dtype == cp.complex128:
            count = result.size * 2  # complex128 has 2 float64 numbers
            nccl_type = nccl.NCCL_FLOAT64
        elif result.dtype == cp.complex64:
            count = result.size * 2  # complex64 has 2 float32 numbers
            nccl_type = nccl.NCCL_FLOAT32
        else:
            raise TypeError(f"Unsupported dtype for NCCL reduce: {result.dtype}")

        comm.reduce(
            result.data.ptr,
            result.data.ptr,
            count,
            nccl_type,
            nccl.NCCL_SUM,
            root,
            stream_ptr,
        )
        return result
    else:
        raise ValueError(f"Unknown reduce method: {method}")


def dense_vector_tn_MPI(qibo_circ, datatype, n_samples=8):
    """Convert qibo circuit to tensornet (TN) format and perform contraction
    using multi node and multi GPU through MPI.

    The conversion is performed by QiboCircuitToEinsum(), after which it
    goes through 2 steps: pathfinder and execution. The pathfinder looks
    at user defined number of samples (n_samples) iteratively to select
    the least costly contraction path. This is sped up with multi
    thread. After pathfinding the optimal path is used in the actual
    contraction to give a dense vector representation of the TN.

    Parameters:
        qibo_circ: The quantum circuit object.
        datatype (str): Either single ("complex64") or double (complex128) precision.
        n_samples(int): Number of samples for pathfinding.

    Returns:
        Dense vector of quantum circuit.
    """
    comm, rank, size, device_id = initialize_mpi()
    operands = get_operands(qibo_circ, datatype, rank, comm)
    network = Network(*operands, options={"device_id": device_id})
    info = compute_optimal_path(network, n_samples, size, comm)
    path, info = network.contract_path(
        optimize={"path": info.path, "slicing": info.slices}
    )
    slices = compute_slices(info, rank, size)
    result = network.contract(slices=slices)
    return reduce_result(result, comm, method="MPI"), rank


def dense_vector_tn_nccl(qibo_circ, datatype, n_samples=8):
    """Convert qibo circuit to tensornet (TN) format and perform contraction
    using multi node and multi GPU through NCCL.

    The conversion is performed by QiboCircuitToEinsum(), after which it
    goes through 2 steps: pathfinder and execution. The pathfinder looks
    at user defined number of samples (n_samples) iteratively to select
    the least costly contraction path. This is sped up with multi
    thread. After pathfinding the optimal path is used in the actual
    contraction to give a dense vector representation of the TN.

    Parameters:
        qibo_circ: The quantum circuit object.
        datatype (str): Either single ("complex64") or double (complex128) precision.
        n_samples(int): Number of samples for pathfinding.

    Returns:
        Dense vector of quantum circuit.
    """
    comm_mpi, rank, size, device_id = initialize_mpi()
    comm_nccl = initialize_nccl(comm_mpi, rank, size)
    operands = get_operands(qibo_circ, datatype, rank, comm_mpi)
    network = Network(*operands)
    info = compute_optimal_path(network, n_samples, size, comm_mpi)
    path, info = network.contract_path(
        optimize={"path": info.path, "slicing": info.slices}
    )
    slices = compute_slices(info, rank, size)
    result = network.contract(slices=slices)
    return reduce_result(result, comm_nccl, method="NCCL"), rank


def dense_vector_tn(qibo_circ, datatype):
    """Convert qibo circuit to tensornet (TN) format and perform contraction to
    dense vector.

    Parameters:
        qibo_circ: The quantum circuit object.
        datatype (str): Either single ("complex64") or double (complex128) precision.

    Returns:
        Dense vector of quantum circuit.
    """
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    return contract(*myconvertor.state_vector_operands())


def expectation_tn_nccl(qibo_circ, datatype, observable, n_samples=8):
    """Convert qibo circuit to tensornet (TN) format and perform contraction to
    expectation of given Pauli string using multi node and multi GPU through
    NCCL.

    The conversion is performed by QiboCircuitToEinsum(), after which it
    goes through 2 steps: pathfinder and execution. The
    pauli_string_pattern is used to generate the pauli string
    corresponding to the number of qubits of the system. The pathfinder
    looks at user defined number of samples (n_samples) iteratively to
    select the least costly contraction path. This is sped up with multi
    thread. After pathfinding the optimal path is used in the actual
    contraction to give an expectation value.

    Parameters:
        qibo_circ: The quantum circuit object.
        datatype (str): Either single ("complex64") or double (complex128) precision.
        pauli_string_pattern(str): pauli string pattern.
        n_samples(int): Number of samples for pathfinding.

    Returns:
        Expectation of quantum circuit due to pauli string.
    """

    comm_mpi, rank, size, device_id = initialize_mpi()

    comm_nccl = initialize_nccl(comm_mpi, rank, size)

    observable = check_observable(observable, qibo_circ.nqubits)

    ham_gate_map = extract_gates_and_qubits(observable)

    if rank == 0:
        myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)

    exp = 0
    for each_ham in ham_gate_map:
        ham_gates = get_ham_gates(each_ham)
        # Perform circuit conversion
        if rank == 0:
            operands = myconvertor.expectation_operands(ham_gates)
        else:
            operands = None

        operands = comm_mpi.bcast(operands, root=0)

        network = Network(*operands)

        # Compute the path on all ranks with 8 samples for hyperoptimization. Force slicing to enable parallel contraction.
        info = compute_optimal_path(network, n_samples, size, comm_mpi)

        # Recompute path with the selected optimal settings
        path, info = network.contract_path(
            optimize={"path": info.path, "slicing": info.slices}
        )

        slices = compute_slices(info, rank, size)

        # Contract the group of slices the process is responsible for.
        result = network.contract(slices=slices)

        # Sum the partial contribution from each process on root.
        result = reduce_result(result, comm_nccl, method="NCCL", root=0)

        exp += result

    return exp, rank


def expectation_tn_MPI(qibo_circ, datatype, observable, n_samples=8):
    """Convert qibo circuit to tensornet (TN) format and perform contraction to
    expectation of given Pauli string using multi node and multi GPU through
    MPI.

    The conversion is performed by QiboCircuitToEinsum(), after which it
    goes through 2 steps: pathfinder and execution. The
    pauli_string_pattern is used to generate the pauli string
    corresponding to the number of qubits of the system. The pathfinder
    looks at user defined number of samples (n_samples) iteratively to
    select the least costly contraction path. This is sped up with multi
    thread. After pathfinding the optimal path is used in the actual
    contraction to give an expectation value.

    Parameters:
        qibo_circ: The quantum circuit object.
        datatype (str): Either single ("complex64") or double (complex128) precision.
        pauli_string_pattern(str): pauli string pattern.
        n_samples(int): Number of samples for pathfinding.

    Returns:
        Expectation of quantum circuit due to pauli string.
    """
    # Initialize MPI and device
    comm, rank, size, device_id = initialize_mpi()

    observable = check_observable(observable, qibo_circ.nqubits)

    ham_gate_map = extract_gates_and_qubits(observable)

    if rank == 0:
        myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    exp = 0
    for each_ham in ham_gate_map:
        ham_gates = get_ham_gates(each_ham)
        # Perform circuit conversion
        # Perform circuit conversion
        if rank == 0:
            operands = myconvertor.expectation_operands(ham_gates)
        else:
            operands = None

        operands = comm.bcast(operands, root=0)

        # Create network object.
        network = Network(*operands, options={"device_id": device_id})

        # Compute optimal contraction path
        info = compute_optimal_path(network, n_samples, size, comm)

        # Set path and slices.
        path, info = network.contract_path(
            optimize={"path": info.path, "slicing": info.slices}
        )

        # Compute slice range for each rank
        slices = compute_slices(info, rank, size)

        # Perform contraction
        result = network.contract(slices=slices)

        # Sum the partial contribution from each process on root.
        result = reduce_result(result, comm, method="MPI", root=0)

        if rank == 0:
            exp += result

    return exp, rank


def expectation_tn(qibo_circ, datatype, observable):
    """Convert qibo circuit to tensornet (TN) format and perform contraction to
    expectation of given Pauli string.

    Parameters:
        qibo_circ: The quantum circuit object.
        datatype (str): Either single ("complex64") or double (complex128) precision.
        pauli_string_pattern(str): pauli string pattern.

    Returns:
        Expectation of quantum circuit due to pauli string.
    """
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)

    observable = check_observable(observable, qibo_circ.nqubits)

    ham_gate_map = extract_gates_and_qubits(observable)
    exp = 0
    for each_ham in ham_gate_map:
        ham_gates = get_ham_gates(each_ham)
        expectation_operands = myconvertor.expectation_operands(ham_gates)
        exp += contract(*expectation_operands)
    return exp


def dense_vector_mps(qibo_circ, gate_algo, datatype):
    """Convert qibo circuit to matrix product state (MPS) format and perform
    contraction to dense vector.

    Parameters:
        qibo_circ: The quantum circuit object.
        gate_algo(dict): Dictionary for SVD and QR settings.
        datatype (str): Either single ("complex64") or double (complex128) precision.

    Returns:
        Dense vector of quantum circuit.
    """
    myconvertor = QiboCircuitToMPS(qibo_circ, gate_algo, dtype=datatype)
    mps_helper = MPSContractionHelper(myconvertor.num_qubits)

    return mps_helper.contract_state_vector(
        myconvertor.mps_tensors, {"handle": myconvertor.handle}
    )

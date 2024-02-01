import multiprocessing

import cupy as cp
from cupy.cuda.runtime import getDeviceCount
from cuquantum import contract
from cupy.cuda.runtime import getDeviceCount
import cupy as cp

from qibotn.mps_contraction_helper import MPSContractionHelper
from qibotn.QiboCircuitConvertor import QiboCircuitToEinsum
from qibotn.QiboCircuitToMPS import QiboCircuitToMPS


def dense_vector_tn(qibo_circ, datatype):
    """Convert qibo circuit to tensornet (TN) format and perform contraction to dense vector."""
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    return contract(*myconvertor.state_vector_operands())


def expectation_pauli_tn(qibo_circ, datatype, pauli_string_pattern):
    """Convert qibo circuit to tensornet (TN) format and perform contraction to expectation of given Pauli string."""
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    return contract(
        *myconvertor.expectation_operands(
            pauli_string_gen(qibo_circ.nqubits, pauli_string_pattern)
        )
    )


def dense_vector_tn_MPI(qibo_circ, datatype, n_samples=8):
    """Convert qibo circuit to tensornet (TN) format and perform contraction using multi node and multi GPU through MPI.
    The conversion is performed by QiboCircuitToEinsum(), after which it goes through 2 steps: pathfinder and execution.
    The pathfinder looks at user defined number of samples (n_samples) iteratively to select the least costly contraction path. This is sped up with multi thread.
    After pathfinding the optimal path is used in the actual contraction to give a dense vector representation of the TN.
    """

    from mpi4py import MPI
    from cuquantum import Network

    root = 0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    device_id = rank % getDeviceCount()

    # Perform circuit conversion
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)

    operands = myconvertor.state_vector_operands()

    # Assign the device for each process.
    device_id = rank % getDeviceCount()

    # Create network object.
    network = Network(*operands, options={"device_id": device_id})

    # Compute the path on all ranks with 8 samples for hyperoptimization. Force slicing to enable parallel contraction.
    path, info = network.contract_path(
        optimize={"samples": n_samples, "slicing": {"min_slices": max(32, size)}}
    )

    # Select the best path from all ranks.
    opt_cost, sender = comm.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)

    # Broadcast info from the sender to all other ranks.
    info = comm.bcast(info, sender)

    # Set path and slices.
    path, info = network.contract_path(
        optimize={"path": info.path, "slicing": info.slices}
    )

    # Calculate this process's share of the slices.
    num_slices = info.num_slices
    chunk, extra = num_slices // size, num_slices % size
    slice_begin = rank * chunk + min(rank, extra)
    slice_end = (
        num_slices if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
    )
    slices = range(slice_begin, slice_end)

    # Contract the group of slices the process is responsible for.
    result = network.contract(slices=slices)

    # Sum the partial contribution from each process on root.
    result = comm.reduce(sendobj=result, op=MPI.SUM, root=root)

    return result, rank


def dense_vector_tn_nccl(qibo_circ, datatype, n_samples=8):
    """Convert qibo circuit to tensornet (TN) format and perform contraction using multi node and multi GPU through NCCL.
    The conversion is performed by QiboCircuitToEinsum(), after which it goes through 2 steps: pathfinder and execution.
    The pathfinder looks at user defined number of samples (n_samples) iteratively to select the least costly contraction path. This is sped up with multi thread.
    After pathfinding the optimal path is used in the actual contraction to give a dense vector representation of the TN.
    """
    from mpi4py import MPI
    from cuquantum import Network
    from cupy.cuda import nccl

    root = 0
    comm_mpi = MPI.COMM_WORLD
    rank = comm_mpi.Get_rank()
    size = comm_mpi.Get_size()

    device_id = rank % getDeviceCount()

    cp.cuda.Device(device_id).use()

    # Set up the NCCL communicator.
    nccl_id = nccl.get_unique_id() if rank == root else None
    nccl_id = comm_mpi.bcast(nccl_id, root)
    comm_nccl = nccl.NcclCommunicator(size, nccl_id, rank)

    # Perform circuit conversion
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    operands = myconvertor.state_vector_operands()

    network = Network(*operands)

    # Compute the path on all ranks with 8 samples for hyperoptimization. Force slicing to enable parallel contraction.
    path, info = network.contract_path(
        optimize={"samples": n_samples, "slicing": {"min_slices": max(32, size)}}
    )

    # Select the best path from all ranks.
    opt_cost, sender = comm_mpi.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)

    # Broadcast info from the sender to all other ranks.
    info = comm_mpi.bcast(info, sender)

    # Set path and slices.
    path, info = network.contract_path(
        optimize={"path": info.path, "slicing": info.slices}
    )

    # Calculate this process's share of the slices.
    num_slices = info.num_slices
    chunk, extra = num_slices // size, num_slices % size
    slice_begin = rank * chunk + min(rank, extra)
    slice_end = (
        num_slices if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
    )
    slices = range(slice_begin, slice_end)

    # Contract the group of slices the process is responsible for.
    result = network.contract(slices=slices)

    # Sum the partial contribution from each process on root.
    stream_ptr = cp.cuda.get_current_stream().ptr
    comm_nccl.reduce(
        result.data.ptr,
        result.data.ptr,
        result.size,
        nccl.NCCL_FLOAT64,
        nccl.NCCL_SUM,
        root,
        stream_ptr,
    )

    return result, rank


def expectation_pauli_tn_nccl(qibo_circ, datatype, pauli_string_pattern, n_samples=8):
    """Convert qibo circuit to tensornet (TN) format and perform contraction to expectation of given Pauli string using multi node and multi GPU through NCCL.
    The conversion is performed by QiboCircuitToEinsum(), after which it goes through 2 steps: pathfinder and execution.
    The pauli_string_pattern is used to generate the pauli string corresponding to the number of qubits of the system.
    The pathfinder looks at user defined number of samples (n_samples) iteratively to select the least costly contraction path. This is sped up with multi thread.
    After pathfinding the optimal path is used in the actual contraction to give an expectation value.
    """
    from mpi4py import MPI
    from cuquantum import Network
    from cupy.cuda import nccl

    root = 0
    comm_mpi = MPI.COMM_WORLD
    rank = comm_mpi.Get_rank()
    size = comm_mpi.Get_size()

    device_id = rank % getDeviceCount()

    cp.cuda.Device(device_id).use()

    # Set up the NCCL communicator.
    nccl_id = nccl.get_unique_id() if rank == root else None
    nccl_id = comm_mpi.bcast(nccl_id, root)
    comm_nccl = nccl.NcclCommunicator(size, nccl_id, rank)

    # Perform circuit conversion
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    operands = myconvertor.expectation_operands(
        pauli_string_gen(qibo_circ.nqubits, pauli_string_pattern)
    )

    network = Network(*operands)

    # Compute the path on all ranks with 8 samples for hyperoptimization. Force slicing to enable parallel contraction.
    path, info = network.contract_path(
        optimize={"samples": n_samples, "slicing": {"min_slices": max(32, size)}}
    )

    # Select the best path from all ranks.
    opt_cost, sender = comm_mpi.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)

    # Broadcast info from the sender to all other ranks.
    info = comm_mpi.bcast(info, sender)

    # Set path and slices.
    path, info = network.contract_path(
        optimize={"path": info.path, "slicing": info.slices}
    )

    # Calculate this process's share of the slices.
    num_slices = info.num_slices
    chunk, extra = num_slices // size, num_slices % size
    slice_begin = rank * chunk + min(rank, extra)
    slice_end = (
        num_slices if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
    )
    slices = range(slice_begin, slice_end)

    # Contract the group of slices the process is responsible for.
    result = network.contract(slices=slices)

    # Sum the partial contribution from each process on root.
    stream_ptr = cp.cuda.get_current_stream().ptr
    comm_nccl.reduce(
        result.data.ptr,
        result.data.ptr,
        result.size,
        nccl.NCCL_FLOAT64,
        nccl.NCCL_SUM,
        root,
        stream_ptr,
    )

    return result, rank


def expectation_pauli_tn_MPI(qibo_circ, datatype, pauli_string_pattern, n_samples=8):
    """Convert qibo circuit to tensornet (TN) format and perform contraction to expectation of given Pauli string using multi node and multi GPU through MPI.
    The conversion is performed by QiboCircuitToEinsum(), after which it goes through 2 steps: pathfinder and execution.
    The pauli_string_pattern is used to generate the pauli string corresponding to the number of qubits of the system.
    The pathfinder looks at user defined number of samples (n_samples) iteratively to select the least costly contraction path. This is sped up with multi thread.
    After pathfinding the optimal path is used in the actual contraction to give an expectation value.
    """
    from mpi4py import MPI  # this line initializes MPI
    from cuquantum import Network

    root = 0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    device_id = rank % getDeviceCount()

    # Perform circuit conversion
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)

    operands = myconvertor.expectation_operands(
        pauli_string_gen(qibo_circ.nqubits, pauli_string_pattern)
    )

    # Assign the device for each process.
    device_id = rank % getDeviceCount()

    # Create network object.
    network = Network(*operands, options={"device_id": device_id})

    # Compute the path on all ranks with 8 samples for hyperoptimization. Force slicing to enable parallel contraction.
    path, info = network.contract_path(
        optimize={"samples": n_samples, "slicing": {"min_slices": max(32, size)}}
    )

    # Select the best path from all ranks.
    opt_cost, sender = comm.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)

    # Broadcast info from the sender to all other ranks.
    info = comm.bcast(info, sender)

    # Set path and slices.
    path, info = network.contract_path(
        optimize={"path": info.path, "slicing": info.slices}
    )

    # Calculate this process's share of the slices.
    num_slices = info.num_slices
    chunk, extra = num_slices // size, num_slices % size
    slice_begin = rank * chunk + min(rank, extra)
    slice_end = (
        num_slices if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
    )
    slices = range(slice_begin, slice_end)

    # Contract the group of slices the process is responsible for.
    result = network.contract(slices=slices)

    # Sum the partial contribution from each process on root.
    result = comm.reduce(sendobj=result, op=MPI.SUM, root=root)

    return result, rank


def dense_vector_mps(qibo_circ, gate_algo, datatype):
    """Convert qibo circuit to matrix product state (MPS) format and perform contraction to dense vector."""
    myconvertor = QiboCircuitToMPS(qibo_circ, gate_algo, dtype=datatype)
    mps_helper = MPSContractionHelper(myconvertor.num_qubits)

    return mps_helper.contract_state_vector(
        myconvertor.mps_tensors, {"handle": myconvertor.handle}
    )


def pauli_string_gen(nqubits, pauli_string_pattern):
    """Used internally to generate the string based on given pattern and number of qubit.
    Example: pattern: "XZ", number of qubit: 7, output = XZXZXZX
    """
    if nqubits <= 0:
        return "Invalid input. N should be a positive integer."

    result = ""

    for i in range(nqubits):
        char_to_add = pauli_string_pattern[i % len(pauli_string_pattern)]
        result += char_to_add
    return result

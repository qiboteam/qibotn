import multiprocessing

import cupy as cp
from cupy.cuda.runtime import getDeviceCount
from cuquantum import contract
from cuquantum import cutensornet as cutn

from qibotn.mps_contraction_helper import MPSContractionHelper
from qibotn.QiboCircuitConvertor import QiboCircuitToEinsum
from qibotn.QiboCircuitToMPS import QiboCircuitToMPS


def eval(qibo_circ, datatype):
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    return contract(*myconvertor.state_vector_operands())


def eval_expectation(qibo_circ, datatype):
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    return contract(
        *myconvertor.expectation_operands(PauliStringGen(qibo_circ.nqubits))
    )


def eval_tn_MPI_2(qibo_circ, datatype, n_samples=8):
    from mpi4py import MPI  # this line initializes MPI
    import socket
    from cuquantum import Network

    # Get the hostname
    # hostname = socket.gethostname()

    root = 0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: Start",mem_avail, "rank =",rank, "hostname =",hostname)
    device_id = rank % getDeviceCount()

    # Perform circuit conversion
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft convetor",mem_avail, "rank =",rank)
    operands = myconvertor.state_vector_operands()
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft operand interleave",mem_avail, "rank =",rank)

    # Broadcast the operand data.
    # operands = comm.bcast(operands, root)

    # Assign the device for each process.
    device_id = rank % getDeviceCount()

    # dev = cp.cuda.Device(device_id)
    # free_mem, total_mem = dev.mem_info
    # print("Mem free: ",free_mem, "Total mem: ",total_mem, "rank =",rank)

    # Create network object.
    network = Network(*operands, options={"device_id": device_id})

    # Compute the path on all ranks with 8 samples for hyperoptimization. Force slicing to enable parallel contraction.
    path, info = network.contract_path(
        optimize={"samples": 8, "slicing": {"min_slices": max(32, size)}}
    )
    # print(f"Process {rank} has the path with the  FLOP count {info.opt_cost}.")

    # Select the best path from all ranks.
    opt_cost, sender = comm.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)

    # if rank == root:
    #    print(f"Process {sender} has the path with the lowest FLOP count {opt_cost}.")

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

    # print(f"Process {rank} is processing slice range: {slices}.")

    # Contract the group of slices the process is responsible for.
    result = network.contract(slices=slices)
    # print(f"Process {rank} result shape is : {result.shape}.")
    # print(f"Process {rank} result size is : {result.nbytes}.")

    # Sum the partial contribution from each process on root.
    result = comm.reduce(sendobj=result, op=MPI.SUM, root=root)

    return result, rank


def eval_tn_nccl(qibo_circ, datatype, n_samples=8):
    from mpi4py import MPI  # this line initializes MPI
    import socket
    from cuquantum import Network
    from cupy.cuda import nccl

    # Get the hostname
    # hostname = socket.gethostname()

    root = 0
    comm_mpi = MPI.COMM_WORLD
    rank = comm_mpi.Get_rank()
    size = comm_mpi.Get_size()
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: Start",mem_avail, "rank =",rank, "hostname =",hostname)
    device_id = rank % getDeviceCount()

    cp.cuda.Device(device_id).use()

    # Set up the NCCL communicator.
    nccl_id = nccl.get_unique_id() if rank == root else None
    nccl_id = comm_mpi.bcast(nccl_id, root)
    comm_nccl = nccl.NcclCommunicator(size, nccl_id, rank)

    # Perform circuit conversion
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft convetor",mem_avail, "rank =",rank)
    operands = myconvertor.state_vector_operands()
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft operand interleave",mem_avail, "rank =",rank)

    network = Network(*operands)

    # Compute the path on all ranks with 8 samples for hyperoptimization. Force slicing to enable parallel contraction.
    path, info = network.contract_path(
        optimize={"samples": 8, "slicing": {"min_slices": max(32, size)}}
    )

    # print(f"Process {rank} has the path with the  FLOP count {info.opt_cost}.")

    # Select the best path from all ranks.
    opt_cost, sender = comm_mpi.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)

    # if rank == root:
    #    print(f"Process {sender} has the path with the lowest FLOP count {opt_cost}.")

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

    # print(f"Process {rank} is processing slice range: {slices}.")

    # Contract the group of slices the process is responsible for.
    result = network.contract(slices=slices)
    # print(f"Process {rank} result shape is : {result.shape}.")
    # print(f"Process {rank} result size is : {result.nbytes}.")

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


def eval_tn_nccl_expectation(qibo_circ, datatype, n_samples=8):
    from mpi4py import MPI  # this line initializes MPI
    import socket
    from cuquantum import Network
    from cupy.cuda import nccl

    # Get the hostname
    # hostname = socket.gethostname()

    root = 0
    comm_mpi = MPI.COMM_WORLD
    rank = comm_mpi.Get_rank()
    size = comm_mpi.Get_size()
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: Start",mem_avail, "rank =",rank, "hostname =",hostname)
    device_id = rank % getDeviceCount()

    cp.cuda.Device(device_id).use()

    # Set up the NCCL communicator.
    nccl_id = nccl.get_unique_id() if rank == root else None
    nccl_id = comm_mpi.bcast(nccl_id, root)
    comm_nccl = nccl.NcclCommunicator(size, nccl_id, rank)

    # Perform circuit conversion
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft convetor",mem_avail, "rank =",rank)
    operands = myconvertor.expectation_operands(PauliStringGen(qibo_circ.nqubits))

    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft operand interleave",mem_avail, "rank =",rank)

    network = Network(*operands)

    # Compute the path on all ranks with 8 samples for hyperoptimization. Force slicing to enable parallel contraction.
    path, info = network.contract_path(
        optimize={"samples": 8, "slicing": {"min_slices": max(32, size)}}
    )

    # print(f"Process {rank} has the path with the  FLOP count {info.opt_cost}.")

    # Select the best path from all ranks.
    opt_cost, sender = comm_mpi.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)

    # if rank == root:
    #    print(f"Process {sender} has the path with the lowest FLOP count {opt_cost}.")

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

    # print(f"Process {rank} is processing slice range: {slices}.")

    # Contract the group of slices the process is responsible for.
    result = network.contract(slices=slices)
    # print(f"Process {rank} result shape is : {result.shape}.")
    # print(f"Process {rank} result size is : {result.nbytes}.")

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


def eval_tn_MPI_2_expectation(qibo_circ, datatype, n_samples=8):
    from mpi4py import MPI  # this line initializes MPI
    import socket
    from cuquantum import Network

    # Get the hostname
    # hostname = socket.gethostname()

    root = 0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: Start",mem_avail, "rank =",rank, "hostname =",hostname)
    device_id = rank % getDeviceCount()

    # Perform circuit conversion
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft convetor",mem_avail, "rank =",rank)
    operands = myconvertor.expectation_operands(PauliStringGen(qibo_circ.nqubits))
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft operand interleave",mem_avail, "rank =",rank)

    # Broadcast the operand data.
    # operands = comm.bcast(operands, root)

    # Assign the device for each process.
    device_id = rank % getDeviceCount()

    # dev = cp.cuda.Device(device_id)
    # free_mem, total_mem = dev.mem_info
    # print("Mem free: ",free_mem, "Total mem: ",total_mem, "rank =",rank)

    # Create network object.
    network = Network(*operands, options={"device_id": device_id})

    # Compute the path on all ranks with 8 samples for hyperoptimization. Force slicing to enable parallel contraction.
    path, info = network.contract_path(
        optimize={"samples": 8, "slicing": {"min_slices": max(32, size)}}
    )
    # print(f"Process {rank} has the path with the  FLOP count {info.opt_cost}.")

    # Select the best path from all ranks.
    opt_cost, sender = comm.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)

    # if rank == root:
    #    print(f"Process {sender} has the path with the lowest FLOP count {opt_cost}.")

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

    # print(f"Process {rank} is processing slice range: {slices}.")

    # Contract the group of slices the process is responsible for.
    result = network.contract(slices=slices)
    # print(f"Process {rank} result shape is : {result.shape}.")
    # print(f"Process {rank} result size is : {result.nbytes}.")

    # Sum the partial contribution from each process on root.
    result = comm.reduce(sendobj=result, op=MPI.SUM, root=root)

    return result, rank


def eval_tn_MPI_expectation(qibo_circ, datatype, n_samples=8):
    from mpi4py import MPI  # this line initializes MPI
    import socket

    # Get the hostname
    # hostname = socket.gethostname()

    ncpu_threads = multiprocessing.cpu_count() // 2

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: Start",mem_avail, "rank =",rank, "hostname =",hostname)
    device_id = rank % getDeviceCount()
    cp.cuda.Device(device_id).use()

    handle = cutn.create()
    network_opts = cutn.NetworkOptions(handle=handle, blocking="auto")
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft network opts",mem_avail, "rank =",rank)
    cutn.distributed_reset_configuration(handle, *cutn.get_mpi_comm_pointer(comm))
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft distributed reset config",mem_avail, "rank =",rank)
    # Perform circuit conversion
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    operands_interleave = myconvertor.expectation_operands(
        PauliStringGen(qibo_circ.nqubits)
    )
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft convetor",mem_avail, "rank =",rank)
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft operand interleave",mem_avail, "rank =",rank)

    # Pathfinder: To search for the optimal path. Optimal path are assigned to path and info attribute of the network object.
    network = cutn.Network(*operands_interleave, options=network_opts)
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft cutn.Network(*operands_interleave,",mem_avail, "rank =",rank)
    path, opt_info = network.contract_path(
        optimize={
            "samples": n_samples,
            "threads": ncpu_threads,
            "slicing": {"min_slices": max(16, size)},
        }
    )
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft contract path",mem_avail, "rank =",rank)
    # Execution: To execute the contraction using the optimal path found previously
    # print("opt_cost",opt_info.opt_cost, "Process =",rank)

    num_slices = opt_info.num_slices  # Andy
    chunk, extra = num_slices // size, num_slices % size  # Andy
    slice_begin = rank * chunk + min(rank, extra)  # Andy
    slice_end = (
        num_slices if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
    )  # Andy
    slices = range(slice_begin, slice_end)  # Andy
    result = network.contract(slices=slices)
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft contract",mem_avail, "rank =",rank)
    cutn.destroy(handle)

    return result, rank


def eval_tn_MPI(qibo_circ, datatype, n_samples=8):
    """Convert qibo circuit to tensornet (TN) format and perform contraction using multi node and multi GPU through MPI.
    The conversion is performed by QiboCircuitToEinsum(), after which it goes through 2 steps: pathfinder and execution.
    The pathfinder looks at user defined number of samples (n_samples) iteratively to select the least costly contraction path. This is sped up with multi thread.
    After pathfinding the optimal path is used in the actual contraction to give a dense vector representation of the TN.
    """

    from mpi4py import MPI  # this line initializes MPI
    import socket

    # Get the hostname
    # hostname = socket.gethostname()

    ncpu_threads = multiprocessing.cpu_count() // 2

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: Start",mem_avail, "rank =",rank, "hostname =",hostname)
    device_id = rank % getDeviceCount()
    cp.cuda.Device(device_id).use()

    handle = cutn.create()
    network_opts = cutn.NetworkOptions(handle=handle, blocking="auto")
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft network opts",mem_avail, "rank =",rank)
    cutn.distributed_reset_configuration(handle, *cutn.get_mpi_comm_pointer(comm))
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft distributed reset config",mem_avail, "rank =",rank)
    # Perform circuit conversion
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft convetor",mem_avail, "rank =",rank)
    operands_interleave = myconvertor.state_vector_operands()
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft operand interleave",mem_avail, "rank =",rank)

    # Pathfinder: To search for the optimal path. Optimal path are assigned to path and info attribute of the network object.
    network = cutn.Network(*operands_interleave, options=network_opts)
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft cutn.Network(*operands_interleave,",mem_avail, "rank =",rank)
    network.contract_path(
        optimize={
            "samples": n_samples,
            "threads": ncpu_threads,
            "slicing": {"min_slices": max(16, size)},
        }
    )
    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft contract path",mem_avail, "rank =",rank)
    # Execution: To execute the contraction using the optimal path found previously
    # print("opt_cost",opt_info.opt_cost, "Process =",rank)

    """
    path, opt_info = network.contract_path(optimize={"samples": n_samples, "threads": ncpu_threads, 'slicing': {'min_slices': max(16, size)}})

    num_slices = opt_info.num_slices#Andy
    chunk, extra = num_slices // size, num_slices % size#Andy
    slice_begin = rank * chunk + min(rank, extra)#Andy
    slice_end = num_slices if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)#Andy
    slices = range(slice_begin, slice_end)#Andy
    result = network.contract(slices=slices)
    """
    result = network.contract()

    # mem_avail = cp.cuda.Device().mem_info[0]
    # print("Mem avail: aft contract",mem_avail, "rank =",rank)
    cutn.destroy(handle)

    return result, rank


def eval_mps(qibo_circ, gate_algo, datatype):
    myconvertor = QiboCircuitToMPS(qibo_circ, gate_algo, dtype=datatype)
    mps_helper = MPSContractionHelper(myconvertor.num_qubits)

    return mps_helper.contract_state_vector(
        myconvertor.mps_tensors, {"handle": myconvertor.handle}
    )


def PauliStringGen(nqubits):
    if nqubits <= 0:
        return "Invalid input. N should be a positive integer."

    # characters = 'IXYZ'
    characters = "XXXZ"

    result = ""

    for i in range(nqubits):
        char_to_add = characters[i % len(characters)]
        result += char_to_add

    return result

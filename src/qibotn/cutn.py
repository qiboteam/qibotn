from qibotn.QiboCircuitConvertor import QiboCircuitToEinsum
from cuquantum import contract
from cuquantum import cutensornet as cutn
import multiprocessing
from cupy.cuda.runtime import getDeviceCount
import cupy as cp


def eval(qibo_circ, datatype):
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    return contract(*myconvertor.state_vector_operands())


def eval_tn_MPI(qibo_circ, datatype, n_samples=8):
    """Convert qibo circuit to tensornet (TN) format and perform contraction using multi node and multi GPU through MPI.
    The conversion is performed by QiboCircuitToEinsum(), after which it goes through 2 steps: pathfinder and execution.
    The pathfinder looks at user defined number of samples (n_samples) iteratively to select the least costly contraction path. This is sped up with multi thread.
    After pathfinding the optimal path is used in the actual contraction to give a dense vector representation of the TN.
    """

    from mpi4py import MPI  # this line initializes MPI

    ncpu_threads = multiprocessing.cpu_count() // 2

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    device_id = rank % getDeviceCount()
    cp.cuda.Device(device_id).use()

    handle = cutn.create()
    cutn.distributed_reset_configuration(handle, *cutn.get_mpi_comm_pointer(comm))
    network_opts = cutn.NetworkOptions(handle=handle, blocking="auto")

    # Perform circuit conversion
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    operands_interleave = myconvertor.state_vector_operands()

    # Pathfinder: To search for the optimal path. Optimal path are assigned to path and info attribute of the network object.
    network = cutn.Network(*operands_interleave, options=network_opts)
    network.contract_path(optimize={"samples": n_samples, "threads": ncpu_threads})

    # Execution: To execute the contraction using the optimal path found previously
    result = network.contract()

    cutn.destroy(handle)

    return result, rank

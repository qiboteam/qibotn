from qibotn.QiboCircuitConvertor import QiboCircuitToEinsum
from cuquantum import contract
from cuquantum import cutensornet as cutn
from mpi4py import MPI  # this line initializes MPI
import multiprocessing
from cupy.cuda.runtime import getDeviceCount


def eval(qibo_circ, datatype):
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    return contract(*myconvertor.state_vector_operands())


def eval_tn_MPI(qibo_circ, datatype):
    
    ncpu_threads = multiprocessing.cpu_count() // 2
    n_samples = 8
    
    root = 0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    device_id = rank % getDeviceCount()
    cp.cuda.Device(device_id).use()
    
    handle = cutn.create()
    cutn.distributed_reset_configuration(handle, *cutn.get_mpi_comm_pointer(comm))
    network_opts = cutn.NetworkOptions(handle=handle, blocking="auto")

    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    operands_interleave = myconvertor.state_vector_operands()
    
    network = cutn.Network(*operands_interleave, options=network_opts)
    network.contract_path(optimize={'samples': n_samples, 'threads': ncpu_threads}) # Calculate optimal path, returns path and info
    
    result = network.contract()
    
    cutn.destroy(handle)
    
    if rank == root:
        return result, rank


if __name__ == "__main__":
    
    from qibo.models import QFT 
    import cupy as cp
    import numpy as np
    
    num_qubits  = 10
    swaps = True
    circ_qibo = QFT(num_qubits, swaps)
    
    dtype="complex128"
    sv_mpi, rank = eval_tn_MPI(circ_qibo, dtype)
    
    if rank == 0:
        sv_reference = eval(circ_qibo, dtype)
        state_vec = np.array(circ_qibo())
        print(f"State vector difference: {abs(sv_mpi-sv_reference).max():0.3e}")
        assert cp.allclose(sv_mpi, sv_reference)
        assert cp.allclose(sv_mpi.flatten(), state_vec)
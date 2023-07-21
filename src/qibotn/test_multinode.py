import os
import sys
from timeit import default_timer as timer
import numpy as np
import cupy as cp
import cuquantum
from cuquantum import cutensornet as cutn
from qibo import gates
from qibo.models import QFT
from mpi4py import MPI  # this line initializes MPI
from cupy.cuda.runtime import getDeviceCount
from qibotn.QiboCircuitConvertor import QiboCircuitToEinsum

def qibo_qft(nqubits, swaps):
    circ_qibo = QFT(nqubits, swaps)
    state_vec = np.array(circ_qibo())
    return circ_qibo, state_vec

args = sys.argv

if len(args) < 2:
    print("Usage: python script.py [nqubits] ")
    sys.exit(1)
    
nqubits = int(args[1])

root = 0
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
device_id = rank % getDeviceCount()
cp.cuda.Device(device_id).use()
#print("Andy: Rank ", rank," size ", size, 'Device count',getDeviceCount())

# Check if the env var is set
if not "CUTENSORNET_COMM_LIB" in os.environ:
    raise RuntimeError("need to set CUTENSORNET_COMM_LIB to the path of the MPI wrapper library")

if not os.path.isfile(os.environ["CUTENSORNET_COMM_LIB"]):
    raise RuntimeError("CUTENSORNET_COMM_LIB does not point to the path of the MPI wrapper library")

datatype = 'complex128'
qibo_circ = QFT(nqubits)
myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)

# Bind the communicator to the library handle
handle = cutn.create()
cutn.distributed_reset_configuration(
    handle, *cutn.get_mpi_comm_pointer(comm)
)

if rank == root:
    start = timer()
    
result = cuquantum.contract(*myconvertor.state_vector_operands(), options={'device_id' : device_id, 'handle': handle})

if rank == root:
    end = timer()

# Check correctness.
if rank == root:
    #(qibo_circ, result_sv) =  qibo_qft(nqubits, swaps=True)
    time = end - start
    #print("Does the cuQuantum parallel contraction result match the cupy.einsum result?", cp.allclose(result.flatten(), result_sv))
    print("nqubit", nqubits, "time taken = ", time, 's')
    
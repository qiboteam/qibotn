import qibo
#import qibotn.cutn as cutn
from cuquantum import cutensornet as cutn

from qibo import gates
from qibo.models import Circuit, QFT
import numpy as np
from mpi4py import MPI  # this line initializes MPI
import cupy as cp
from cupy.cuda.runtime import getDeviceCount
from qibotn.QiboCircuitConvertor import QiboCircuitToEinsum
import cuquantum

def qibo_qft(nqubits, swaps):
    circ_qibo = QFT(nqubits, swaps)
    state_vec = np.array(circ_qibo())
    return circ_qibo, state_vec

print("QiboTN")

root = 0
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
print("Andy: Rank ", rank," size ", size)
# Assign the device for each process.
device_id = rank % getDeviceCount()
cp.cuda.Device(device_id).use()

datatype = 'complex128'
nqubits = 10
'''
qibo_circ = Circuit(nqubits)
qibo_circ.add(gates.H(0))
#qibo_circ.add(gates.CZ(3,4))
qibo_circ.add(gates.CZ(2,4))
#qibo_circ.add(gates.CNOT(0,4))
#qibo_circ.add(gates.SWAP(0,4))
qibo_circ.add(gates.H(2))
qibo_circ.add(gates.H(4))
'''
qibo_circ = QFT(nqubits)

'''
expr = 'ehl,gj,edhg,bif,d,c,k,iklj,cf,a->ba'
shapes = [(8, 2, 5), (5, 7), (8, 8, 2, 5), (8, 6, 3), (8,), (6,), (5,), (6, 5, 5, 7), (6, 3), (3,)]
print("Andy: expr =",expr)
if rank == root:
    operands = [cp.random.rand(*shape) for shape in shapes]
else:
    operands = [cp.empty(shape) for shape in shapes]
'''

myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
expr, mode_label, q_frontier, operands = myconvertor.state_vector()
shapes = [tensor.shape for tensor in operands]
print("expr ", expr)
print("Operands ", operands)
print("Shape", shapes)
# Set the operand data on root. Since we use the buffer interface APIs offered by mpi4py for communicating array
#  objects, we can directly use device arrays (cupy.ndarray, for example) as we assume mpi4py is built against
#  a CUDA-aware MPI.
if rank != root:
    operands = [cp.empty(shape,dtype="complex128") for shape in shapes]

'''    
if rank == root:
    operands = [cp.random.rand(*shape) for shape in shapes]
    print("Operands random", operands)

else:
    operands = [cp.empty(shape) for shape in shapes]
'''

for operand in operands:
    print("Is CUPY array? ", cp.get_array_module(operand), " Operand size = ", operand.nbytes)

for operand in operands:
    comm.Bcast(operand, root)

# Bind the communicator to the library handle
handle = cutn.create()
print("Andy cutn.create()")
print("Andy ", cutn.get_mpi_comm_pointer(comm))
cutn.distributed_reset_configuration(
    handle, *cutn.get_mpi_comm_pointer(comm)
)
print("Andy cutn.distributed_reset_configuration")

operands_interleave = myconvertor.get_interleave_format( mode_label, q_frontier, operands)
print("new function interkeave ", operands_interleave)
print("Ori function interleave", myconvertor.state_vector_operands())

result = cuquantum.contract(*operands_interleave, options={'device_id' : device_id, 'handle': handle})
#result = cuquantum.contract(expr, *operands, options={'device_id' : device_id, 'handle': handle})

'''

# Create a new GPU buffer for verification
result_cp = cp.empty_like(result)

# Sum the partial contribution from each process on root, with GPU
if rank == root:
    comm.Reduce(sendbuf=MPI.IN_PLACE, recvbuf=result_cp, op=MPI.SUM, root=root)
else:
    comm.Reduce(sendbuf=result_cp, recvbuf=None, op=MPI.SUM, root=root)
'''
# Check correctness.
if rank == root:
    #operands = myconvertor.state_vector_operands()
    #result_cp = cp.einsum(*operands, optimize=True)
    #result_cp = np.einsum(*operands, optimize=True)
    (qibo_circ, result_sv) =  qibo_qft(nqubits, swaps=True)
    print("Does the cuQuantum parallel contraction result match the cupy.einsum result?", cp.allclose(result.flatten(), result_sv))


'''
result_tn = cutn.eval(qibo_circ, datatype)

qibo.set_backend(backend="qibojit", platform="numpy")
(qibo_circ, result_sv) = qibo_qft(nqubits, swaps=True)
#print(result_tn)
#print(result_sv)

assert np.allclose(
        result_sv, result_tn.flatten()), "Resulting dense vectors do not match"
'''
from qibotn.QiboCircuitConvertor import QiboCircuitToEinsum
from cuquantum import contract
from MpsHelper import MPSHelper
import cuquantum 
from cuquantum import cutensornet as cutn
import cupy as cp
import numpy as np

def eval(qibo_circ, datatype):
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    return contract(*myconvertor.state_vector_operands())

def eval_mps(qibo_circ, datatype):
    #Create MPS
    cutensornet.create()
    return contract()


if __name__ == "__main__":
    print("cuTensorNet-vers:", cutn.get_version())
    dev = cp.cuda.Device()  # get current device
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    print("===== device info ======")
    print("GPU-name:", props["name"].decode())
    print("GPU-clock:", props["clockRate"])
    print("GPU-memoryClock:", props["memoryClockRate"])
    print("GPU-nSM:", props["multiProcessorCount"])
    print("GPU-major:", props["major"])
    print("GPU-minor:", props["minor"])
    print("========================")
    
    data_type = cuquantum.cudaDataType.CUDA_C_64F
    compute_type = cuquantum.ComputeType.COMPUTE_64F
    num_sites = 16
    phys_extent = 2
    max_virtual_extent = 12
    
    ## we initialize the MPS state as a product state |000...000>
    initial_state = []
    for i in range(num_sites):
        # we create dummpy indices for MPS tensors on the boundary for easier bookkeeping
        # we'll use Fortran layout throughout this example
        tensor = cp.zeros((1,2,1), dtype=np.complex128, order="F")
        tensor[0,0,0] = 1.0
        initial_state.append(tensor)
    
    mps_helper = MPSHelper(num_sites, phys_extent, max_virtual_extent, initial_state, data_type, compute_type)

    ##################################
    # Setup options for gate operation
    ##################################
    
    abs_cutoff = 1e-2
    rel_cutoff = 1e-2
    renorm = cutn.TensorSVDNormalization.L2
    partition = cutn.TensorSVDPartition.UV_EQUAL
    mps_helper.set_svd_config(abs_cutoff, rel_cutoff, renorm, partition)

    gate_algo = cutn.GateSplitAlgo.REDUCED
    mps_helper.set_gate_algorithm(gate_algo)
    
    #####################################
    # Workspace estimation and allocation
    #####################################

    free_mem, total_mem = dev.mem_info
    worksize = free_mem *.7
    required_workspace_size = mps_helper.compute_max_workspace_sizes()
    work = cp.cuda.alloc(worksize)
    print(f"Maximal workspace size requried: {required_workspace_size / 1024 ** 3:.3f} GB")
    mps_helper.set_workspace(work, required_workspace_size)
    

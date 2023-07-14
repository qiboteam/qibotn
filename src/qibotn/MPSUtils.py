import cupy as cp
from cuquantum.cutensornet.experimental import contract_decompose
from cuquantum import contract

def get_initial_mps(num_qubits, dtype='complex128'):
    """
    Generate the MPS with an initial state of |00...00> 
    """
    state_tensor = cp.asarray([1, 0], dtype=dtype).reshape(1,2,1)
    mps_tensors = [state_tensor] * num_qubits
    return mps_tensors

def mps_site_right_swap(
    mps_tensors, 
    i, 
    algorithm=None, 
    options=None
):
    """
    Perform the swap operation between the ith and i+1th MPS tensors.
    """
    # contraction followed by QR decomposition
    a, _, b = contract_decompose('ipj,jqk->iqj,jpk', *mps_tensors[i:i+2], algorithm=algorithm, options=options)
    mps_tensors[i:i+2] = (a, b)
    return mps_tensors

def apply_gate(
    mps_tensors, 
    gate, 
    qubits, 
    algorithm=None, 
    options=None
):
    """
    Apply the gate operand to the MPS tensors in-place.
    
    Args:
        mps_tensors: A list of rank-3 ndarray-like tensor objects. 
            The indices of the ith tensor are expected to be the bonding index to the i-1 tensor, 
            the physical mode, and then the bonding index to the i+1th tensor.
        gate: A ndarray-like tensor object representing the gate operand. 
            The modes of the gate is expected to be output qubits followed by input qubits, e.g, 
            ``A, B, a, b`` where ``a, b`` denotes the inputs and ``A, B`` denotes the outputs. 
        qubits: A sequence of integers denoting the qubits that the gate is applied onto.
        algorithm: The contract and decompose algorithm to use for gate application. 
            Can be either a `dict` or a `ContractDecomposeAlgorithm`.
        options: Specify the contract and decompose options. 
    
    Returns:
        The updated MPS tensors.
    """
    
    n_qubits = len(qubits)
    if n_qubits == 1:
        # single-qubit gate
        i = qubits[0]
        mps_tensors[i] = contract('ipj,qp->iqj', mps_tensors[i], gate, options=options) # in-place update
    elif n_qubits == 2:
        # two-qubit gate
        i, j = qubits
        if i > j:
            # swap qubits order
            return apply_gate(mps_tensors, gate.transpose(1,0,3,2), (j, i), algorithm=algorithm, options=options)
        elif i+1 == j:
            # two adjacent qubits
            a, _, b = contract_decompose('ipj,jqk,rspq->irj,jsk', *mps_tensors[i:i+2], gate, algorithm=algorithm, options=options)
            mps_tensors[i:i+2] = (a, b) # in-place update
        else:
            # non-adjacent two-qubit gate
            # step 1: swap i with i+1
            mps_site_right_swap(mps_tensors, i, algorithm=algorithm, options=options)
            # step 2: apply gate to (i+1, j) pair. This amounts to a recursive swap until the two qubits are adjacent
            apply_gate(mps_tensors, gate, (i+1, j), algorithm=algorithm, options=options) 
            # step 3: swap back i and i+1
            mps_site_right_swap(mps_tensors, i, algorithm=algorithm, options=options)
    else:
        raise NotImplementedError("Only one- and two-qubit gates supported")
    return mps_tensors
import argparse
import quimb as qu
import quimb.tensor as qtn
import numpy as np
import re, copy

import qibo
from qibo.models import QFT as qibo_qft

from timeit import default_timer as timer

def get_gate_params(operation):
    if "h " in operation:
        qbit_no = [int(re.findall(r'\d+', operation)[0])]
        qbit_no.insert(0, "H")
    elif "x " in operation:
        qbit_no = [int(re.findall(r'\d+', operation)[0])]
        qbit_no.insert(0, "X")
    elif "y " in operation:
        qbit_no = [int(re.findall(r'\d+', operation)[0])]
        qbit_no.insert(0, "Y")
    elif "z " in operation:
        qbit_no = [int(re.findall(r'\d+', operation)[0])]
        qbit_no.insert(0, "Z")
    elif "s " in operation:
        qbit_no = [int(re.findall(r'\d+', operation)[0])]
        qbit_no.insert(0, "S")
    elif "t " in operation:
        qbit_no = [int(re.findall(r'\d+', operation)[0])]
        qbit_no.insert(0, "T")
    elif "cu1" in operation:
        lamda = float('.'.join(re.findall(r'\b\d+(?:[Ee][+-]?\d+)?', \
            operation.split(" ")[0])))
        qbit_no = re.findall(r'\d+', operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no[0:0] = ["CU1", lamda]
    elif "cu2" in operation:
        angles = re.findall(r'\b\d+(?:[Ee][+-]?\d+)?',operation.split(" ")[0])
        phi = float('.'.join(angles[0:2]))
        lamba = float('.'.join(angles[2:]))
        qbit_no = re.findall(r'\d+', operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no[0:0] = ["CU2", phi, lamda]
    elif "cu3" in operation:
        angles = re.findall(r'\b\d+(?:[Ee][+-]?\d+)?',operation.split(" ")[0])
        theta = float('.'.join(angles[0:2]))
        phi = float('.'.join(angles[2:4]))
        lamba = float('.'.join(angles[4:]))
        qbit_no = re.findall(r'\d+', operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no[0:0] = ["CU3", theta, phi, lamda]       
    elif " cx " in operation:
        qbit_no = re.findall(r'\d+', operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no.insert(0, "CX")
    elif " cy " in operation:
        qbit_no = re.findall(r'\d+', operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no.insert(0, "CY") 
    elif " cz " in operation:
        qbit_no = re.findall(r'\d+', operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no.insert(0, "CZ")
    elif " ccx " in operation:
        qbit_no = re.findall(r'\d+', operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no.insert(0, "CCX")
    elif " ccy " in operation:
        qbit_no = re.findall(r'\d+', operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no.insert(0, "CCY")
    elif " ccz " in operation:
        qbit_no = re.findall(r'\d+', operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no.insert(0, "CCZ")
    elif " rx " in operation:
        theta = float('.'.join(re.findall(r'\b\d+(?:[Ee][+-]?\d+)?', \
            operation.split(" ")[0])))
        qbit_no = [int(re.findall(r'\d+', operation)[0])]
        qbit_no[0:0] = ["RX", theta]
    elif "^ry " in operation:
        theta = float('.'.join(re.findall(r'\b\d+(?:[Ee][+-]?\d+)?', \
            operation.split(" ")[0])))
        qbit_no = [int(re.findall(r'\d+', operation)[0])]
        qbit_no[0:0] = ["RY", theta]
    elif "^rz " in operation:
        theta = float('.'.join(re.findall(r'\b\d+(?:[Ee][+-]?\d+)?', \
            operation.split(" ")[0])))
        qbit_no = [int(re.findall(r'\d+', operation)[0])]
        qbit_no[0:0] = ["RZ", theta]
    elif "^rzz " in operation:
        theta = float('.'.join(re.findall(r'\b\d+(?:[Ee][+-]?\d+)?', \
            operation.split(" ")[0])))
        qbit_no = re.findall(r'\d+', operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no[0:0] = ["RZZ", theta]
    elif "^u1 " in operation:
        lamda = float('.'.join(re.findall(r'\b\d+(?:[Ee][+-]?\d+)?', \
            operation.split(" ")[0])))
        qbit_no = [int(re.findall(r'\d+', operation)[0])]
        qbit_no[0:0] = ["U1", lamda]
    elif "^u2 " in operation:
        angles = re.findall(r'\b\d+(?:[Ee][+-]?\d+)?',operation.split(" ")[0])
        phi = float('.'.join(angles[0:2]))
        lamba = float('.'.join(angles[2:]))
        qbit_no = int(re.findall(r'\d+', operation)[0])
        qbit_no[0:0] = ["U2", phi, lamda]
    elif "^u3 " in operation:
        angles = re.findall(r'\b\d+(?:[Ee][+-]?\d+)?',operation.split(" ")[0])
        theta = float('.'.join(angles[0:2]))
        phi = float('.'.join(angles[2:4]))
        lamba = float('.'.join(angles[4:]))
        qbit_no = int(re.findall(r'\d+', operation)[0])
        qbit_no[0:0] = ["U3", theta, phi, lamda]
    else:
        assert("Unsupported gate")

    return qbit_no


def get_gate_functions(qasm_str, start_idx):
    func_list = []
    result = []
    idx_inc = 0
    for line in qasm_str[start_idx:]:
        if "gate " in line:
            result = re.findall("[^,\s()]+", line)
        elif result and "{" not in line and "}" not in line:
            params = get_gate_params(line)
            func_list.append(*params)
        elif "}" in line:
            print("Returning the list")
            print(func_list)
            return func_list, idx_incsss
        idx_inc += 1

def qasm_QFT(nqubits:int, qasm_str:str, with_swaps: bool = True, psi0 = None):
    circ = qtn.Circuit(nqubits, psi0 = psi0)

    qasm_str = qasm_str.split('\n')
    for idx, line in enumerate(qasm_str):
        command = line.split(" ")[0]
        if re.search("include|//|OPENQASM", command):
            continue
        elif "qreg" in command:
            nbits = int(re.findall(r'\d+', line)[0])
            assert(nbits == nqubits)
        elif "swap" in command:
            break
        elif "gate" in command:      # TODO: Complete gate handling
            gate_func, increment = get_gate_functions(qasm_str, idx)
            pass
        elif "barrier" in command:   # TODO: Complete barrier handling
            pass
        elif "measure" in command:   # TODO: Complete measure handling
            pass
        else:
            params = get_gate_params(line)
            circ.apply_gate(*params)

    if with_swaps:
        for i in range(nqubits // 2): # TODO: Ignore the barrier indices? 
            circ.apply_gate('SWAP', i, nqubits - i - 1)

    return circ


def eval_QI_qft(nqubits, backend='numpy', qibo_backend='qibojit', \
    with_swaps=True):
    # backend (quimb): numpy, cupy, jax. Passed to ``opt_einsum``.
    # qibo_backend: qibojit, qibotf, tensorflow, numpy
    
    # generate random statevector as initial state
    init_state = np.random.random(2 ** nqubits) + 1j * \
        np.random.random(2 ** nqubits)
    init_state = init_state / np.sqrt((np.abs(init_state)**2).sum())
    init_state_quimb = copy.deepcopy(init_state)
    
    # Qibo circuit
    # qibo.set_backend(backend=qibo_backend, platform="numba")
    qibo.set_backend(backend=qibo_backend, platform="numpy")

    start = timer()
    circ_qibo = qibo_qft(nqubits, with_swaps)
    amplitudes_reference = np.array(circ_qibo(init_state))
    end = timer()
    print("qibo time is " + str(end-start))
    qasm_circ = circ_qibo.to_qasm()

 
    #####################################################################
    # Quimb circuit
    qu.core.pnjit()
    ## convert vector to MPS
    dims = tuple(2*np.ones(nqubits, dtype=int))
    start = timer() 
    init_state_MPS = qtn.tensor_1d.MatrixProductState.from_dense(init_state_quimb, dims)
    end = timer()
    MPS_time = end-start

    
    # construct quimb qft circuit
    start = timer()
    circ_quimb = qasm_QFT(nqubits, qasm_circ, with_swaps, psi0=init_state_MPS)

    interim = circ_quimb.psi.full_simplify(seq="DRC")

    result = interim.to_dense(backend=backend)
    amplitudes = result.flatten()
    end = timer()
    quimb_qft_time = end-start
    print("quimb time is " + str(quimb_qft_time))
    assert(np.allclose(amplitudes,amplitudes_reference,atol=1e-06))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=10, type=int,
                        help="Number of quibits in the circuits.")

    args = parser.parse_args()

    print("Testing for %d nqubits" % (args.nqubits))
    result = eval_QI_qft(args.nqubits)

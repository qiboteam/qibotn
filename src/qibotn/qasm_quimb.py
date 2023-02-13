import re
import copy
from timeit import default_timer as timer

import numpy as np
import quimb as qu
import quimb.tensor as qtn

import qibo
from qibo.models import QFT as qibo_qft


def get_gate_params(operation):
    if "h " in operation:
        qbit_no = [int(re.findall(r"\d+", operation)[0])]
        qbit_no.insert(0, "H")
    elif "x " in operation:
        qbit_no = [int(re.findall(r"\d+", operation)[0])]
        qbit_no.insert(0, "X")
    elif "y " in operation:
        qbit_no = [int(re.findall(r"\d+", operation)[0])]
        qbit_no.insert(0, "Y")
    elif "z " in operation:
        qbit_no = [int(re.findall(r"\d+", operation)[0])]
        qbit_no.insert(0, "Z")
    elif "s " in operation:
        qbit_no = [int(re.findall(r"\d+", operation)[0])]
        qbit_no.insert(0, "S")
    elif "t " in operation:
        qbit_no = [int(re.findall(r"\d+", operation)[0])]
        qbit_no.insert(0, "T")
    elif "cu1" in operation:
        lambda_ = float(
            ".".join(re.findall(r"\b\d+(?:[Ee][+-]?\d+)?", operation.split(" ")[0]))
        )
        qbit_no = re.findall(r"\d+", operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no[0:0] = ["CU1", lambda_]
    elif "cu2" in operation:
        angles = re.findall(r"\b\d+(?:[Ee][+-]?\d+)?", operation.split(" ")[0])
        phi = float(".".join(angles[0:2]))
        lambda_ = float(".".join(angles[2:]))
        qbit_no = re.findall(r"\d+", operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no[0:0] = ["CU2", phi, lambda_]
    elif "cu3" in operation:
        angles = re.findall(r"\b\d+(?:[Ee][+-]?\d+)?", operation.split(" ")[0])
        theta = float(".".join(angles[0:2]))
        phi = float(".".join(angles[2:4]))
        lambda_ = float(".".join(angles[4:]))
        qbit_no = re.findall(r"\d+", operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no[0:0] = ["CU3", theta, phi, lambda_]
    elif " cx " in operation:
        qbit_no = re.findall(r"\d+", operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no.insert(0, "CX")
    elif " cy " in operation:
        qbit_no = re.findall(r"\d+", operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no.insert(0, "CY")
    elif " cz " in operation:
        qbit_no = re.findall(r"\d+", operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no.insert(0, "CZ")
    elif " ccx " in operation:
        qbit_no = re.findall(r"\d+", operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no.insert(0, "CCX")
    elif " ccy " in operation:
        qbit_no = re.findall(r"\d+", operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no.insert(0, "CCY")
    elif " ccz " in operation:
        qbit_no = re.findall(r"\d+", operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no.insert(0, "CCZ")
    elif " rx " in operation:
        theta = float(
            ".".join(re.findall(r"\b\d+(?:[Ee][+-]?\d+)?", operation.split(" ")[0]))
        )
        qbit_no = [int(re.findall(r"\d+", operation)[0])]
        qbit_no[0:0] = ["RX", theta]
    elif "^ry " in operation:
        theta = float(
            ".".join(re.findall(r"\b\d+(?:[Ee][+-]?\d+)?", operation.split(" ")[0]))
        )
        qbit_no = [int(re.findall(r"\d+", operation)[0])]
        qbit_no[0:0] = ["RY", theta]
    elif "^rz " in operation:
        theta = float(
            ".".join(re.findall(r"\b\d+(?:[Ee][+-]?\d+)?", operation.split(" ")[0]))
        )
        qbit_no = [int(re.findall(r"\d+", operation)[0])]
        qbit_no[0:0] = ["RZ", theta]
    elif "^rzz " in operation:
        theta = float(
            ".".join(re.findall(r"\b\d+(?:[Ee][+-]?\d+)?", operation.split(" ")[0]))
        )
        qbit_no = re.findall(r"\d+", operation.split(" ")[1])
        qbit_no = [int(x) for x in qbit_no]
        qbit_no[0:0] = ["RZZ", theta]
    elif "^u1 " in operation:
        lambda_ = float(
            ".".join(re.findall(r"\b\d+(?:[Ee][+-]?\d+)?", operation.split(" ")[0]))
        )
        qbit_no = [int(re.findall(r"\d+", operation)[0])]
        qbit_no[0:0] = ["U1", lambda_]
    elif "^u2 " in operation:
        angles = re.findall(r"\b\d+(?:[Ee][+-]?\d+)?", operation.split(" ")[0])
        phi = float(".".join(angles[0:2]))
        lambda_ = float(".".join(angles[2:]))
        qbit_no = int(re.findall(r"\d+", operation)[0])
        qbit_no[0:0] = ["U2", phi, lambda_]  # pylint: disable=E1137
    elif "^u3 " in operation:
        angles = re.findall(r"\b\d+(?:[Ee][+-]?\d+)?", operation.split(" ")[0])
        theta = float(".".join(angles[0:2]))
        phi = float(".".join(angles[2:4]))
        lambda_ = float(".".join(angles[4:]))
        qbit_no = int(re.findall(r"\d+", operation)[0])
        qbit_no[0:0] = ["U3", theta, phi, lambda_]  # pylint: disable=E1137
    else:
        assert "Unsupported gate"

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
            return func_list, idx_inc
        idx_inc += 1


def qasm_QFT(nqubits: int, qasm_str: str, with_swaps: bool = True, psi0=None):
    circ = qtn.Circuit(nqubits, psi0=psi0)

    qasm_str = qasm_str.split("\n")
    for idx, line in enumerate(qasm_str):
        command = line.split(" ")[0]
        if re.search("include|//|OPENQASM", command):
            continue
        elif "qreg" in command:
            nbits = int(re.findall(r"\d+", line)[0])
            assert nbits == nqubits
        elif "swap" in command:
            break
        elif "gate" in command:  # TODO: Complete gate handling
            gate_func, increment = get_gate_functions(qasm_str, idx)
            pass
        elif "barrier" in command:  # TODO: Complete barrier handling
            pass
        elif "measure" in command:  # TODO: Complete measure handling
            pass
        else:
            params = get_gate_params(line)
            circ.apply_gate(*params)

    if with_swaps:
        for i in range(nqubits // 2):  # TODO: Ignore the barrier indices?
            circ.apply_gate("SWAP", i, nqubits - i - 1)

    return circ


def init_state_tn(nqubits, init_state_sv, tn_lib="quimb"):
    dims = tuple(2 * np.ones(nqubits, dtype=int))

    if tn_lib == "quimb":
        init_state_MPS = qtn.tensor_1d.MatrixProductState.from_dense(
            init_state_sv, dims)
    else:
        # TODO: Add cuquantum later
        assert False, "Unsupported tensor network backend in initilization"

    return init_state_MPS


def tn_circ_eval(nqubits, qasm_circ, init_state, swaps=True, tn_lib="quimb",
        backend='numpy'):
    if tn_lib == "quimb":
        circ_quimb = qasm_QFT(nqubits, qasm_circ, swaps, psi0=init_state)
        interim = circ_quimb.psi.full_simplify(seq="DRC")
        result = interim.to_dense(backend=backend).flatten()
        return result
    else:
        # TODO: Change assert or value. Add cuquantum later
        assert False, "Unsupported tensor network library"


def eval_QI_qft(nqubits, backend="numpy", qibo_backend="qibojit", with_swaps=True):
    # backend (quimb): numpy, cupy, jax. Passed to ``opt_einsum``.
    # qibo_backend: qibojit, qibotf, tensorflow, numpy

    # generate random statevector as initial state
    init_state = np.random.random(2**nqubits) + 1j * np.random.random(2**nqubits)
    init_state = init_state / np.sqrt((np.abs(init_state) ** 2).sum())
    init_state_quimb = copy.deepcopy(init_state)

    # Qibo circuit
    # qibo.set_backend(backend=qibo_backend, platform="numba")
    qibo.set_backend(backend=qibo_backend, platform="numpy")

    start = timer()
    circ_qibo = qibo_qft(nqubits, with_swaps)
    amplitudes_reference = np.array(circ_qibo(init_state))
    end = timer()
    qibo_qft_time = end - start
    qasm_circ = circ_qibo.to_qasm()

    #####################################################################
    # Quimb circuit
    init_state_MPS = init_state_tn(nqubits=nqubits,
                                   init_state_sv=init_state_quimb)

    # construct quimb qft circuit
    start = timer()
    amplitudes = tn_circ_eval(nqubits=nqubits, qasm_circ=qasm_circ,
                              init_state=init_state_MPS, swaps=with_swaps,
                              tn_lib="quimb")
    end = timer()
    quimb_qft_time = end - start
    assert np.allclose(amplitudes, amplitudes_reference, atol=1e-06)

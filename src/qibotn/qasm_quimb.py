import re

import quimb as qu
import quimb.tensor as qtn
import numpy as np


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
            init_state_sv, dims
        )
    else:
        # TODO: Add cuquantum later
        assert False, "Unsupported tensor network backend in initilization"

    return init_state_MPS


def tn_circ_eval(
    nqubits, qasm_circ, init_state, swaps=True, tn_lib="quimb", backend="numpy"
):
    if tn_lib == "quimb":

        circ_quimb = qasm_QFT(nqubits, qasm_circ, swaps, psi0=init_state)
        interim = circ_quimb.psi.full_simplify(seq="DRC")
        result = interim.to_dense(backend=backend).flatten()
        return result
    else:
        # TODO: Change assert or value. Add cuquantum later
        assert False, "Unsupported tensor network library"


def eval_QI_qft(nqubits, qasm_circ, init_state, backend="numpy", swaps=True):
    # backend (quimb): numpy, cupy, jax. Passed to ``opt_einsum``.

    # Quimb circuit
    init_state_mps = init_state_tn(nqubits=nqubits, init_state_sv=init_state)
    amplitudes = tn_circ_eval(
        nqubits=nqubits,
        qasm_circ=qasm_circ,
        init_state=init_state_mps,
        swaps=swaps,
        tn_lib="quimb",
    )
    return amplitudes

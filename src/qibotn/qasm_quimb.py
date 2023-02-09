import re
import copy
from timeit import default_timer as timer

import numpy as np
import quimb.tensor as qtn

import qibo
from qibo.models import QFT as qibo_qft

from .qasm import gate_params


def gate_functions(qasm_str, start_idx):
    func_list = []
    result = []
    idx_inc = 0

    for line in qasm_str[start_idx:]:
        if "gate " in line:
            result = re.findall(r"[^,\s()]+", line)
        elif result and "{" not in line and "}" not in line:
            params = gate_params(line)
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
            gate_func, increment = gate_functions(qasm_str, idx)
            pass
        elif "barrier" in command:  # TODO: Complete barrier handling
            pass
        elif "measure" in command:  # TODO: Complete measure handling
            pass
        else:
            params = gate_params(line)
            circ.apply_gate(params[0], *params[1:])

    if with_swaps:
        for i in range(nqubits // 2):  # TODO: Ignore the barrier indices?
            circ.apply_gate("SWAP", i, nqubits - i - 1)

    return circ


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
    print("qibo time is " + str(end - start))
    qasm_circ = circ_qibo.to_qasm()

    #####################################################################
    # Quimb circuit
    # convert vector to MPS
    dims = tuple(2 * np.ones(nqubits, dtype=int))
    init_state_MPS = qtn.tensor_1d.MatrixProductState.from_dense(init_state_quimb, dims)

    # construct quimb qft circuit
    start = timer()
    circ_quimb = qasm_QFT(nqubits, qasm_circ, with_swaps, psi0=init_state_MPS)

    interim = circ_quimb.psi.full_simplify(seq="DRC")

    result = interim.to_dense(backend=backend)
    amplitudes = result.flatten()
    end = timer()
    quimb_qft_time = end - start
    print("quimb time is " + str(quimb_qft_time))
    assert np.allclose(amplitudes, amplitudes_reference, atol=1e-06)

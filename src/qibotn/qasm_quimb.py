import re
import copy
from dataclasses import dataclass, field
from enum import Enum
from timeit import default_timer as timer
from typing import Callable, Dict, List, Union

import numpy as np
import quimb.tensor as qtn

import qibo
from qibo.models import QFT as qibo_qft

# string manipulators


def extract_ints(string: str):
    return re.findall(r"\d+", string)


def extract_floats(string: str):
    return re.findall(r"\b\d+(?:[Ee][+-]?\d+)?", string)


def parse_float(strings: List[str]):
    return float(".".join(strings))


def chunk(string: str, n: int):
    return string.split(" ")[n]


# parameters extractors


@dataclass
class Parameter:
    extractor: Callable
    preprocess: Callable = lambda x: x
    postprocess: Callable = lambda x: x


class ParameterKind(Enum):
    FirstInt = Parameter(extract_ints, postprocess=lambda x: int(x[0]))
    IntSecond = Parameter(
        extract_ints,
        preprocess=lambda x: chunk(x, 1),
        postprocess=lambda x: int(x[0]),
    )
    FirstFloat = Parameter(
        extract_floats, preprocess=lambda x: chunk(x, 0), postprocess=parse_float
    )

    def extract(self, string):
        v = self.value
        return v.postprocess(v.extractor(v.preprocess(string)))


ParK = ParameterKind


# Gates


@dataclass
class Gate:
    name: str
    tag: str
    parameters: Dict[str, ParameterKind] = field(default_factory=dict)


class GateKind(Enum):
    H = Gate("H", "h ", {"p0": ParK.FirstInt})
    X = Gate("X", "x ", {"p0": ParK.FirstInt})
    Y = Gate("Y", "y ", {"p0": ParK.FirstInt})
    Z = Gate("Z", "z ", {"p0": ParK.FirstInt})
    S = Gate("S", "s ", {"p0": ParK.FirstInt})
    T = Gate("T", "t ", {"p0": ParK.FirstInt})
    CU1 = Gate("CU1", "cu1", {"lambda": ParK.FirstFloat, "p0": ParK.FirstInt})
    CU2 = Gate(
        "CU2",
        "cu2",
        {"phi": ParK.FirstFloat, "lambda": ParK.FirstInt, "p0": ParK.FirstInt},
    )
    CU3 = Gate(
        "CU3",
        "cu3",
        {
            "theta": ParK.FirstFloat,
            "phi": ParK.FirstFloat,
            "lambda": ParK.FirstInt,
            "p0": ParK.FirstInt,
        },
    )
    CX = Gate("CX", " cx ", {"p0": ParK.IntSecond})
    CY = Gate("CY", " cy ", {"p0": ParK.IntSecond})
    CZ = Gate("CZ", " cz ", {"p0": ParK.IntSecond})
    CCX = Gate("CCX", " ccx ", {"p0": ParK.IntSecond})
    CCY = Gate("CCY", " ccy ", {"p0": ParK.IntSecond})
    CCZ = Gate("CCZ", " ccz ", {"p0": ParK.IntSecond})
    RX = Gate("RX", " rx ", {"p0": ParK.FirstFloat, "theta": ParK.FirstInt})
    RY = Gate("RY", "^ry ", {"p0": ParK.FirstFloat, "theta": ParK.FirstInt})
    RZ = Gate("RZ", "^rz ", {"p0": ParK.FirstFloat, "theta": ParK.FirstInt})
    RZZ = Gate("RZZ", "^rzz ", {"p0": ParK.FirstFloat, "theta": ParK.FirstInt})
    U1 = Gate("U1", "^u1 ", {"p0": ParK.FirstFloat, "lambda": ParK.FirstInt})
    U2 = Gate("U2", "^u2 ")
    U3 = Gate("U3", "^u3 ")


def gate_params(operation: str):
    qbit_no: List[Union[int, float, str]] = []

    for kind in GateKind:
        gate = kind.value
        if gate.tag in operation:
            qbit_no.append(gate.name)
            parameters = [par.extract(operation) for par in gate.parameters.values()]
            qbit_no.extend(parameters)
            break
    else:
        assert "Unsupported gate"

    return qbit_no


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

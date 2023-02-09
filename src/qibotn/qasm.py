import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Union

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
class Extractor:
    extractor: Callable
    preprocess: Callable = lambda x: x
    postprocess: Callable = lambda x: x


class Parameter(Enum):
    FirstInt = Extractor(extract_ints, postprocess=lambda x: int(x[0]))
    IntSecond = Extractor(
        extract_ints,
        preprocess=lambda x: chunk(x, 1),
        postprocess=lambda x: int(x[0]),
    )
    FirstFloat = Extractor(
        extract_floats, preprocess=lambda x: chunk(x, 0), postprocess=parse_float
    )

    def extract(self, string):
        ex = self.value
        return ex.postprocess(ex.extractor(ex.preprocess(string)))


Par = Parameter


# Gates


@dataclass
class Gate:
    name: str
    tag: str
    parameters: Dict[str, Parameter] = field(default_factory=dict)


class GateKind(Enum):
    H = Gate("H", "h ", {"p0": Par.FirstInt})
    X = Gate("X", "x ", {"p0": Par.FirstInt})
    Y = Gate("Y", "y ", {"p0": Par.FirstInt})
    Z = Gate("Z", "z ", {"p0": Par.FirstInt})
    S = Gate("S", "s ", {"p0": Par.FirstInt})
    T = Gate("T", "t ", {"p0": Par.FirstInt})
    CU1 = Gate("CU1", "cu1", {"lambda": Par.FirstFloat, "p0": Par.FirstInt})
    CU2 = Gate(
        "CU2",
        "cu2",
        {"phi": Par.FirstFloat, "lambda": Par.FirstInt, "p0": Par.FirstInt},
    )
    CU3 = Gate(
        "CU3",
        "cu3",
        {
            "theta": Par.FirstFloat,
            "phi": Par.FirstFloat,
            "lambda": Par.FirstInt,
            "p0": Par.FirstInt,
        },
    )
    CX = Gate("CX", " cx ", {"p0": Par.IntSecond})
    CY = Gate("CY", " cy ", {"p0": Par.IntSecond})
    CZ = Gate("CZ", " cz ", {"p0": Par.IntSecond})
    CCX = Gate("CCX", " ccx ", {"p0": Par.IntSecond})
    CCY = Gate("CCY", " ccy ", {"p0": Par.IntSecond})
    CCZ = Gate("CCZ", " ccz ", {"p0": Par.IntSecond})
    RX = Gate("RX", " rx ", {"p0": Par.FirstFloat, "theta": Par.FirstInt})
    RY = Gate("RY", "^ry ", {"p0": Par.FirstFloat, "theta": Par.FirstInt})
    RZ = Gate("RZ", "^rz ", {"p0": Par.FirstFloat, "theta": Par.FirstInt})
    RZZ = Gate("RZZ", "^rzz ", {"p0": Par.FirstFloat, "theta": Par.FirstInt})
    U1 = Gate("U1", "^u1 ", {"p0": Par.FirstFloat, "lambda": Par.FirstInt})
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

import argparse
from timeit import default_timer as timer

from qibotn import quimb as qiboquimb
from QiboCircuitConvertor import QiboCircuitToEinsum
from cuquantum import contract
import cupy as cp
from qibo.models import *


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nqubits", default=10, type=int, help="Number of quibits in the circuits."
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    print("Testing for %d nqubits" % (args.nqubits))
    qiboquimb.eval(args.nqubits, args.qasm_circ, args.init_state)


def parser_cuquantum():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nqubits", default=10, type=int, help="Number of quibits in the circuits."
    )

    parser.add_argument(
        "--circuit",
        default="qft",
        type=str,
        help="Type of circuit to use. See README for the list of "
        "available circuits.",
    )

    parser.add_argument(
        "--precision",
        default="complex128",
        type=str,
        help="Numerical precision of the simulation. "
        "Choose between 'complex128' and 'complex64'.",
    )

    return parser.parse_args()


def main_cuquantum(args: argparse.Namespace):
    print("Testing for %d nqubits" % (args.nqubits))
    nqubits = args.nqubits
    circuit_name = args.circuit
    datatype = args.precision
    # Create qibo quibit

    if circuit_name in ("qft", "QFT"):
        circuit = QFT(nqubits)
    else:
        raise NotImplementedError(f"Cannot find circuit {circuit_name}.")

    myconvertor = QiboCircuitToEinsum(circuit, dtype=datatype)

    expression, operands = myconvertor.state_vector()

    start = timer()
    result_qibo = circuit()
    end = timer()
    circuit_eval_time = end - start
    print("Simulation time: Qibo                  =", circuit_eval_time, "s")

    start = timer()
    sv_cutn = contract(expression, *operands)
    end = timer()
    circuit_eval_time = end - start
    print("Simulation time: cuQuantum cuTensorNet =", circuit_eval_time, "s")

    # print(f"is sv in agreement?", cp.allclose(sv_cutn.flatten(), result_qibo.state(numpy=True)))
    assert cp.allclose(sv_cutn.flatten(), result_qibo.state(numpy=True))


if __name__ == "__main__":
    main(parser())

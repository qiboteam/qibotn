import argparse
from timeit import default_timer as timer

import qibotn.quimb
from QiboCircuitConvertor import QiboCircuitToEinsum
from cuquantum import contract
import cupy as cp
from qibo.models import QFT


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nqubits", default=10, type=int, help="Number of quibits in the circuits."
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    print("Testing for %d nqubits" % (args.nqubits))
    qibotn.quimb.eval(args.nqubits, args.qasm_circ, args.init_state)


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


def run_bench(task, label):
    start = timer()
    result = task()
    end = timer()
    circuit_eval_time = end - start
    print(f"Simulation time: {label} = {circuit_eval_time}s")

    return result


def main_cuquantum(args: argparse.Namespace):
    print("Testing for %d nqubits" % (args.nqubits))
    nqubits = args.nqubits
    circuit_name = args.circuit
    datatype = args.precision

    if circuit_name in ("qft", "QFT"):
        circuit = QFT(nqubits)
    else:
        raise NotImplementedError(f"Cannot find circuit {circuit_name}.")

    myconvertor = QiboCircuitToEinsum(circuit, dtype=datatype)
    operands_expression = myconvertor.state_vector()

    result_qibo = run_bench(circuit, "Qibo")
    sv_cutn = run_bench(lambda: contract(*operands_expression), "cuQuantum cuTensorNet")

    # print(f"is sv in agreement?", cp.allclose(sv_cutn.flatten(), result_qibo.state(numpy=True)))
    assert cp.allclose(sv_cutn.flatten(), result_qibo.state(numpy=True))


if __name__ == "__main__":
    main_cuquantum(parser_cuquantum())

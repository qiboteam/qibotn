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


if __name__ == "__main__":
    main(parser())

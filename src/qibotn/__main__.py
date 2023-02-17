import argparse
from qibotn import qasm_quimb


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nqubits", default=10, type=int, help="Number of quibits in the circuits."
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    print("Testing for %d nqubits" % (args.nqubits))
    qasm_quimb.eval_QI_qft(args.nqubits, args.qasm_circ, args.init_state)


if __name__ == "__main__":
    main(parser())

import argparse

import qibotn.quimb


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

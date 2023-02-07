import os

import pytest


@pytest.mark.parametrize("nqubits", [1, 2, 5, 10])
def test_eval(nqubits: int):
    os.environ["QUIMB_NUM_PROCS"] = str(os.cpu_count())
    from qibotn import qasm_quimb

    print(f"Testing for {nqubits} nqubits")
    result = qasm_quimb.eval_QI_qft(nqubits)

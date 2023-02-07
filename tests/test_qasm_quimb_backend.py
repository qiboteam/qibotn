import pytest

from qibotn import qasm_quimb


@pytest.mark.parametrize("nqubits", [1, 2, 5, 10])
def test_eval(nqubits: int):
    print(f"Testing for {nqubits} nqubits")
    result = qasm_quimb.eval_QI_qft(nqubits)

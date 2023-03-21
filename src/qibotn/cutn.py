# from qibotn import quimb as qiboquimb
from QiboCircuitConvertor import QiboCircuitToEinsum
from cuquantum import contract


def eval(qibo_circ):
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype="complex128")
    operands_expression = myconvertor.state_vector()
    results = contract(*operands_expression)
    return results.flatten()

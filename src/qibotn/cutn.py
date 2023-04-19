# from qibotn import quimb as qiboquimb
from qibotn.QiboCircuitConvertor import QiboCircuitToEinsum
from cuquantum import contract


def eval(qibo_circ, datatype):
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    operands_expression = myconvertor.state_vector_operands()
    results = contract(*operands_expression)
    return results

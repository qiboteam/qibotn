# from qibotn import quimb as qiboquimb
from qibotn.QiboCircuitConvertor import QiboCircuitToEinsum
from cuquantum import contract


def eval(qibo_circ, datatype):
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    return contract(*myconvertor.state_vector_operands())

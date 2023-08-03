from qibotn.QiboCircuitConvertor import QiboCircuitToEinsum
from cuquantum import contract

from qibotn.QiboCircuitToMPS import QiboCircuitToMPS
from qibotn.MPSContractionHelper import MPSContractionHelper


def eval(qibo_circ, datatype):
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    return contract(*myconvertor.state_vector_operands())


def eval_mps(qibo_circ, gate_algo, datatype):
    myconvertor = QiboCircuitToMPS(qibo_circ, gate_algo, dtype=datatype)
    mps_helper = MPSContractionHelper(myconvertor.num_qubits)

    return mps_helper.contract_state_vector(
        myconvertor.mps_tensors, myconvertor.options
    )

from qibotn.QiboCircuitConvertor import QiboCircuitToEinsum
from cuquantum import contract
from cuquantum import cutensornet as cutn
import cupy as cp
import numpy as np
from qibo.models import QFT
from qibotn.QiboCircuitToMPS import QiboCircuitToMPS
from qibotn.MPSContractionHelper import MPSContractionHelper


def eval(qibo_circ, datatype):
    myconvertor = QiboCircuitToEinsum(qibo_circ, dtype=datatype)
    return contract(*myconvertor.state_vector_operands())


def eval_mps(qibo_circ, gate_algo, datatype):
    myconvertor = QiboCircuitToMPS(qibo_circ, gate_algo, dtype=datatype)
    mps_helper = MPSContractionHelper(myconvertor.num_qubits)
    sv_mps = mps_helper.contract_state_vector(
        myconvertor.mps_tensors, myconvertor.options)
    return sv_mps


if __name__ == "__main__":
    num_qubits = 25
    swaps = True
    circ_qibo = QFT(num_qubits, swaps)

    exact_gate_algorithm = {'qr_method': False,
                            'svd_method': {'partition': 'UV', 'abs_cutoff': 1e-12}}
    dtype = "complex128"
    sv_mps = eval_mps(circ_qibo, exact_gate_algorithm, dtype)
    sv_reference = eval(circ_qibo, dtype)
    state_vec = np.array(circ_qibo())
    print(f"State vector difference: {abs(sv_mps-sv_reference).max():0.3e}")
    assert cp.allclose(sv_mps, sv_reference)
    assert cp.allclose(sv_mps.flatten(), state_vec)

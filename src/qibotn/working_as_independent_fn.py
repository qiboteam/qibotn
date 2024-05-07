""" TEBD usage : An independent algorithm function"""
import eval_qu as eval
import qibo
from qibo import hamiltonians

qibo.set_backend(backend="qibotn", platform="qutensornet", runcard=None)

tebd_opts = {"dt":1e-4, "hamiltonian": "XXZ", "initial_state": "10101", "tot_time":1}

ham = hamiltonians.XXZ(nqubits=5, dense=False)
circuit= ham.circuit(dt=1e-4)

print(eval.tebd_tn_qu(circuit, tebd_opts))

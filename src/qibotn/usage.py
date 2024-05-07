import qibo
from qibo import hamiltonians, gates, models, Circuit

dt = 1e-4
nqubits = 5

'''computation_settings = {
    "MPI_enabled": False,
    "MPS_enabled": False,
    "NCCL_enabled": False,
    "expectation_enabled": False,
    "TEBD_enabled" : {"dt":dt, "hamiltonian":"XXZ", "initial_state":"10101", "tot_time":1}
}
'''
computation_settings = {
    "MPI_enabled": False,
    "MPS_enabled": False,
    "NCCL_enabled": False,
    "expectation_enabled": False,
    "TEBD_enabled": False
}


qibo.set_backend(backend="qibotn", platform="qutensornet", runcard=computation_settings)

#ham = hamiltonians.XXZ(nqubits=nqubits, dense=False)
#circuit = ham.circuit(dt=dt)

circuit = Circuit(2)
circuit.add(gates.RY(0, theta=0))
circuit.add(gates.RY(1, theta=0))
circuit.add(gates.CNOT(0,1))

result = circuit()
print(result.state())


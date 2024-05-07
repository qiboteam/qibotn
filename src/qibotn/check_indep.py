import qibo
import numpy as np
import quimb.tensor as qtn
import quimb as qmb
from qibo import Circuit, gates, hamiltonians
""" TEBD usage """
import eval_qu as eval
import qibo
from qibo import hamiltonians

dt = 1e-4
nqubits = 5

computation_settings = {
    "MPI_enabled": False,
    "MPS_enabled": False,
    "NCCL_enabled": False,
    "expectation_enabled": False,
    "TEBD_enabled" : {"dt":dt, "hamiltonian":"XXZ", "initial_state":"10101", "tot_time":1}
}

qibo.set_backend(backend="qibotn", platform="qutensornet", runcard=computation_settings)

initial_state = '10101'
tebd_opts = {"dt":1e-4, "hamiltonian": "XXZ", "initial_state": "10101", "tot_time":1}

ham = hamiltonians.XXZ(nqubits=nqubits, dense=False)
circuit= ham.circuit(dt=dt)

result = circuit()
print(result.state())


print(eval.tebd_tn_qu(circuit, tebd_opts, initial_state, nqubits))
'''
# TEBD_opts
dt = 1e-4
init_state = '00'

# Initial MPS State
initial_state = qtn.MPS_computational_state(init_state)

# MPS_opts
mps_opts={"method": "svd", "cutoff": 1e-6, "cutoff_mode": "abs"}

# A basic VQE ansatz: A circuit that approximates the desired quantum state

ansatz = Circuit(2)
ansatz.add(gates.RY(0, theta=0))
ansatz.add(gates.RY(1, theta=0))
ansatz.add(gates.CNOT(0,1))

# Conversion to QASM
qasm = ansatz.to_qasm()

# Build a Quimb MPS from QASM string
circ = qtn.circuit.CircuitMPS if mps_opts else qtn.circuit.Circuit
circ_quimb = circ.from_openqasm2_str(
        qasm, psi0=initial_state, gate_opts=mps_opts
    )

# Final MPS

final_mps = circ_quimb.psi.full_simplify(seq="DRC")
amplitudes = final_mps.to_dense()

print("Final MPS ",type(final_mps)) # Returns quimb.tensor.tensor_1d.MatrixProductState
print("Dense of Final MPS ",amplitudes) # Returns np.ndarray

# Energy(VQE)

# Preparing the hamiltonian for TEBD
ham = hamiltonians.XXZ(2)

# Make the hamiltonian usable
intermediate = ham.circuit(dt=dt)'''

# Emulate TEBD using 3 Exponentiations

#       Exponentiation 1

#exp1 = np.exp(is*D)

#       Exponentiation 2

#exp2 = np.exp(-is*H0)


#       Exponentiation 3

#exp3 = np.exp(-isD)
 
# Apply these exponentiations to the output MPS

# Energy(DBI)

# Run more steps of DBI

# Idea is to show that: Energy(VQE) > Energy(DBI 1) > Energy(DBI 2) in a plot

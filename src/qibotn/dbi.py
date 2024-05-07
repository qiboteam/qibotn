import qibo
import numpy as np
import quimb.tensor as qtn
import quimb as qmb
from qibo import Circuit, gates, hamiltonians

# TEBD_opts
dt = 1e-4
init_state = '00'

# Initial MPS State
initial_state = qtn.MPS_computational_state(init_state)

# MPS_opts
mps_opts={"method": "svd", "cutoff": 1e-6, "cutoff_mode": "abs"}

# A basic VQE circuit

circuit = Circuit(2)
circuit.add(gates.RY(0, theta=0))
circuit.add(gates.RY(1, theta=0))
circuit.add(gates.CNOT(0,1))

# Conversion to QASM
qasm = circuit.to_qasm()

# Build a Quimb MPS from QASM string
circ = qtn.circuit.CircuitMPS if mps_opts else qtn.circuit.Circuit
circ_quimb = circ.from_openqasm2_str(
        qasm, psi0=initial_state, gate_opts=mps_opts
    )

# Final MPS

final_mps = circ_quimb.psi.full_simplify(seq="DRC")
print("Final MPS ",final_mps)
amplitudes = final_mps.to_dense()
print("Dense vector of Final MPS ",amplitudes)


# Energy(VQE)

# Preparing the hamiltonian for TEBD
ham = hamiltonians.XXZ(2)

# Make the hamiltonian usable
intermediate = ham.circuit(dt=dt) 

i=0
terms_lst = ham.terms
for t in terms_lst:
    effective_ham = effective_ham*t.matrix
    i=i+1

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
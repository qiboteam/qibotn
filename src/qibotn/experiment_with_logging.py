# importing necessary packages
import numpy as np
import quimb.tensor as qtn
from qibo import hamiltonians

# tebd opts
dt = 1e-4
nqubits = 5
init_state = "10101"
tot = 1

# casting ham as crt using td
ham = hamiltonians.XXZ(nqubits=nqubits, dense=False)
circuit = ham.circuit(dt=dt)

# openqasm workaround experiments
# print(circuit.decompose()) # decompose doc says it Returns: Circuit that contains only gates that are supported by OpenQASM and has the same effect as the original circuit.

# build initial state
psi0 = qtn.MPS_computational_state(init_state)

# extract symb rep terms of ham
terms_dict = {}
i = 0
list_of_terms = ham.terms
for t in list_of_terms:
    terms_dict.update({None: t.matrix})
    i = i + 1

# build quimb ham and tebd object with ts time range
H = qtn.LocalHam1D(nqubits, H2=terms_dict)
tebd = qtn.TEBD(psi0, H)
ts = np.arange(0, tot, dt)

result = {}
# write evol dense vect at times ts into txt
# file_path = "tebd_log2.txt"
# with open(file_path, 'w') as file:
for t in tebd.at_times(ts, tol=tot):
    result.update({None: t.to_dense()})
# file.write("\n"+str(t.to_dense()))

res = list(result.values())
res = np.array(res[-1])
print(res)
"""# extract the final evol dense vect
with open(file_path, 'r') as file:
        content = file.read()

state_str = (content.rsplit(']]',2)[-2])+"]]"
state = np.array(eval(state_str))

print(state)"""

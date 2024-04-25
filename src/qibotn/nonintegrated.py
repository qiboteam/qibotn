import qibo
from qibo import hamiltonians, gates, models
from qibo import Circuit
import quimb.tensor as qtn
import numpy as np

dt = 1e-4
nqubits = 5 
init_state = "10101"
tot = 1

ham = hamiltonians.XXZ(nqubits=nqubits, dense=False)
circuit = ham.circuit(dt=dt)

psi0 = qtn.MPS_computational_state(init_state)

terms_dict = {}
i=0
list_of_terms = ham.terms
for t in list_of_terms:
    terms_dict.update({None: t.matrix})
    i=i+1

H = qtn.LocalHam1D(nqubits, H2=terms_dict)
tebd = qtn.TEBD(psi0, H)
ts = np.arange(0, tot, dt)

file_path = "tebd_log.txt"
with open(file_path, 'w') as file:
    for t in tebd.at_times(ts, tol=tot):
        file.write("\n"+str(t.to_dense()))

with open(file_path, 'r') as file:
        content = file.read()

state_str = (content.rsplit(']]',2)[-2])+"]]"
state = np.array(eval(state_str))

print(state)




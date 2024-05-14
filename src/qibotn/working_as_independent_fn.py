""" TEBD usage : An independent algorithm function"""

# import eval_qu as eval
import qibo
from qibo import hamiltonians

qibo.set_backend(backend="qibotn", platform="qutensornet", runcard=None)

tebd_opts = {"dt": 1e-4, "hamiltonian": "XXZ", "initial_state": "10101", "tot_time": 1}

ham = hamiltonians.TFIM(nqubits=2, dense=False)
circuit = ham.circuit(dt=1e-4)

# print(tebd.tebd_quimb(circuit, tebd_opts))
"""Trying circuit based."""

"""print(circuit.unitary())
u = circuit.unitary()
import numpy as np
u = np.array(u)
print(u)
print(u.shape)

import quimb.tensor as qtn
import quimb

circ_cls = qtn.circuit.CircuitMPS

from qiskit import QuantumCircuit
import qiskit.qasm2 as qq

num_qubits = 5
qc = QuantumCircuit(num_qubits)
qc.unitary(u, qubits=range(num_qubits))

from qiskit import transpile

target_basis = ['rx', 'ry', 'rz', 'h', 'cx', 'x', 'y', 'z']
decomposed = transpile(qc,
                       basis_gates=target_basis,
                       optimization_level=0)

qasm = qq.dumps(decomposed)
print(qasm)

initial_state = '00000'
initial_state = qtn.MPS_computational_state(initial_state)

mps_opts = {"method": "svd", "cutoff": 1e-6, "cutoff_mode": "abs"}

circ_cls = qtn.circuit.CircuitMPS
circ_quimb = circ_cls.from_openqasm2_str(
        qasm, psi0=initial_state, gate_opts=mps_opts
    )

interim = circ_quimb.psi.full_simplify(seq="DRC")
amplitudes = interim.to_dense()
print(amplitudes)


#ar = np.zeros((31,31))


#tensor1 = qtn.Tensor(u, inds=ar)
#print(tensor1)
#circ_quimb = qtn.CircuitMPS.from_gates(gates=circuit.unitary(), cutoff=1e-6)
#print(circ_quimb)

#circ_quimb = qtn.CircuitMPS
#circ_quimb.get_uni(u)

#u = np.Tensor(u)
#t = qtn.Tensor(u)
#tn = qtn.TensorNetwork(u)

import qibotn.circuit_to_mps as ev
"""
"""Import numpy as np dt=1e-4.

np.set_printoptions(precision=8)

U = circuit.unitary()
H = np.divide((np.log(U)),-1*dt)

np.nan_to_num(H)
print(H)

# Then build a quimb ham with this

import quimb.tensor as qtn
h = qtn.ham_1d_heis(5)

print(h.terms.values())

arr=[]
for d in h.terms.values():
    arr.append(d*dt)

U = np.exp(arr)
print("unitary ",U)
"""

"""
import quimb.tensor as qtn
from qibo import gates, Circuit
from numpy import pi

vqe_circ = Circuit(2)
vqe_circ.add(gates.RY(0, theta=0))
vqe_circ.add(gates.RY(1, theta=0))
vqe_circ.add(gates.CNOT(0,1))

qasm = vqe_circ.to_qasm()
mps_opts = {"method": "svd", "cutoff": 1e-6, "cutoff_mode": "abs"}
initial_state = '00'

initial_state = qtn.MPS_computational_state(initial_state)

circ_cls = qtn.circuit.CircuitMPS
circ_quimb = circ_cls.from_openqasm2_str(
        qasm, psi0=initial_state, gate_opts=mps_opts
)

interim = circ_quimb.psi.full_simplify(seq="DRC")
amplitudes = interim.to_dense()

initial_state = qtn.MPS_computational_state('10')

ham = qtn.ham_1d_heis(2)

tebd = qtn.TEBD(initial_state, ham)
print(tebd)

tot_time = 1e-3
dt = 1e-4
ts = np.arange(0, tot_time, dt)
states = {}
for t in tebd.at_times(ts, dt=dt):
    states.update({None: t.to_dense()})

state = np.array(list(states.values()))[-1]

print("State ",state)
"""
"""Interim = circ_quimb.psi.full_simplify(seq="DRC") amplitudes =
interim.to_dense()

print(amplitudes*state)
"""

"""
# reverse engineer, from circuit -> unitary -> circuit -> qasm -> quimb mps circuit???
unitary = circuit.unitary()

import quimb.tensor as qtn

from qiskit import QuantumCircuit, QuantumRegister
import qiskit.qasm3 as qq
import qiskit
circ = QuantumCircuit(2)
circ.unitary(unitary,[0,1])
print(circ)

from qiskit import transpile
transpiled_circ = transpile(circ, basis_gates = ['rx','ry','rz','cx'])

qasm = qq.dumps(transpiled_circ)
print(qasm)

mps_opts = {"method": "svd", "cutoff": 1e-6, "cutoff_mode": "abs"}
initial_state = '00'

initial_state = qtn.MPS_computational_state(initial_state)

circ_cls = qtn.circuit.CircuitMPS
circ_quimb = circ_cls.from_openqasm2_str(
        qasm, psi0=initial_state, gate_opts=mps_opts
    )

interim = circ_quimb.psi.full_simplify(seq="DRC")
amplitudes = interim.to_dense()

print(amplitudes)
"""

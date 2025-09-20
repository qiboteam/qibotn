import numpy as np
import jax
from qibo.backends import construct_backend
from qibo import Circuit, gates, hamiltonians
from qibo.symbols import Z, X, Y

# construct qibotn backend
quimb_backend = construct_backend(backend="qibotn", platform="quimb")

quimb_backend.setup_backend_specifics(
    qimb_backend="jax", 
    optimizer='auto-hq'
    )

quimb_backend.configure_tn_simulation(
    max_bond_dimension=10
)


# define Hamiltonian
form = 0.5 * Z(0) * Z(1) +- 1.5 *  X(0) * Z(2) + Z(3)
hamiltonian = hamiltonians.SymbolicHamiltonian(form)


# define circuit
def build_circuit(nqubits, nlayers):
    """Construct a more complex Qibo parametric quantum circuit without CNOT gates."""
    circ = Circuit(nqubits)
    for layer in range(nlayers):
        for q in range(nqubits):
            circ.add(gates.RY(q=q, theta=0.))
            circ.add(gates.RZ(q=q, theta=0.))
            circ.add(gates.RX(q=q, theta=0.))
        # Add controlled rotations and SWAPs for entanglement
        for q in range(nqubits - 1):
            circ.add(gates.CNOT(q, q + 1))
            circ.add(gates.SWAP(q, q + 1))
    circ.add(gates.M(*range(nqubits)))
    return circ

nqubits = 6
circuit = build_circuit(nqubits=nqubits, nlayers=3)

def f(params):
    circuit.set_parameters(params)
    return quimb_backend.expectation(
        circuit=circuit,
        observable=hamiltonian,
    )

parameters = np.random.uniform(-np.pi, np.pi, size=len(circuit.get_parameters()))
print(jax.value_and_grad(f)(parameters))

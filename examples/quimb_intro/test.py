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
    """Construct a Qibo parametric quantum circuit."""
    circ = Circuit(nqubits)
    for _ in range(nlayers):
        for q in range(nqubits):
            circ.add(gates.RY(q=q, theta=0.))
            circ.add(gates.RZ(q=q, theta=0.))
        [circ.add(gates.CZ(q % nqubits, (q + 1) % nqubits)) for q in range(nqubits)]
    circ.add(gates.M(*range(nqubits)))
    return circ

nqubits = 4
circuit = build_circuit(nqubits=nqubits, nlayers=3)

quimb_circuit = quimb_backend._qibo_circuit_to_quimb(circuit)

def f(params):
    circuit.set_parameters(params)
    return quimb_backend.expectation(
        circuit=circuit,
        observable=hamiltonian,
    )

parameters = np.random.uniform(-np.pi, np.pi, size=len(circuit.get_parameters()))
print(jax.value_and_grad(f)(parameters))

import numpy as np
import quimb.tensor as qtn


def init_state_tn(nqubits, init_state_sv):
    """Create a matrix product state directly from a dense vector.

    Args:
        nqubits (int): Total number of qubits in the circuit.
        init_state_sv (list): Initial state in the dense vector form.

    Returns:
        list: Matrix product state representation of the dense vector.
    """

    dims = tuple(2 * np.ones(nqubits, dtype=int))

    return qtn.tensor_1d.MatrixProductState.from_dense(init_state_sv, dims)


def dense_vector_tn_qu(qasm: str, initial_state, mps_opts, backend="numpy"):
    """Evaluate circuit in QASM format with Quimb.

    Args:
        qasm (str): QASM program.
        initial_state (list): Initial state in the dense vector form. If ``None`` the default ``|00...0>`` state is used.
        mps_opts (dict): Parameters to tune the gate_opts for mps settings in ``class quimb.tensor.circuit.CircuitMPS``.
        backend (str):  Backend to perform the contraction with, e.g. ``numpy``, ``cupy``, ``jax``. Passed to ``opt_einsum``.

    Returns:
        list: Amplitudes of final state after the simulation of the circuit.
    """

    if initial_state is not None:
        nqubits = int(np.log2(len(initial_state)))
        initial_state = init_state_tn(nqubits, initial_state)

    circ_cls = qtn.circuit.CircuitMPS if mps_opts else qtn.circuit.Circuit
    circ_quimb = circ_cls.from_openqasm2_str(
        qasm, psi0=initial_state, gate_opts=mps_opts
    )

    interim = circ_quimb.psi.full_simplify(seq="DRC")
    amplitudes = interim.to_dense(backend=backend)

    return amplitudes



def expectation_qu(qasm: str, pauli_string_pattern, initial_state, mps_opts, backend="numpy"):

    qasm_mod, nqubits = modify_qasm(qasm)
    if initial_state is not None:
        initial_state = init_state_tn(nqubits, initial_state)

    circ_cls = qtn.circuit.CircuitMPS if mps_opts else qtn.circuit.Circuit
    circ_quimb = circ_cls.from_openqasm2_str(
        qasm, psi0=initial_state, gate_opts=mps_opts
    )

    obs = pauli_string_gen(nqubits-1, pauli_string_pattern)

    opt = ctg.ReusableHyperOptimizer(
    # just do a few runs
        max_repeats=32,
    # only use the basic greedy optimizer ...
        methods=['greedy'],
    # ... but pair it with reconfiguration
        reconf_opts={},
    # just uniformly sample the space
        optlib='random',
    # terminate search if contraction is cheap
        max_time='rate:1e6',
    # account for both flops and write - usually wise for practical performance
        minimize='combo-64',
    # persist paths found in here
        directory='cotengra_cache_eco',
    )
    expectation = circ_quimb.local_expectation(obs,  where=list(range(nqubits-1)), optimize=opt, 
        simplify_sequence ="DRC")
  
    return expectation

def modify_qasm(qasm_circ):
    lines =qasm_circ.split("\n")
    qasm_circ_mod =[]
    while lines:
        line = lines.pop(0).strip()
        sta = re.compile(r"qreg\s+(\w+)\s*\[(\d+)\];")
        match = sta.match(line)
        if match:
            name, nqubits = match.groups()
            qasm_circ_mod.append(f"qreg q[{int(nqubits)+1}];")
        else:
            qasm_circ_mod.append(line)
    qasm_circ_mod = "\n".join(qasm_circ_mod)
    return qasm_circ_mod, int(nqubits)+1

def pauli_string_gen(nqubits, pauli_string_pattern):
    """Used internally to generate the string based on given pattern and number
    of qubit.

    Parameters:
        nqubits(int): Number of qubits of Quantum Circuit
        pauli_string_pattern(str): Strings representing sequence of pauli gates.

    Returns:
        String representation of the actual pauli string from the pattern.

    Example: pattern: "XZ", number of qubit: 7, output = XZXZXZX
    """
    if nqubits <= 0:
        return "Invalid input. N should be a positive integer."

    result = ""

    for i in range(nqubits):
        char_to_add = pauli_string_pattern[i % len(pauli_string_pattern)]
        result += char_to_add
    

    for i, c in enumerate(result):
        if (i==0):
            obs = qu.pauli(c)
        else:
            obs = obs&qu.pauli(c)
    return obs

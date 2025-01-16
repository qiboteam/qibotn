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


def dense_vector_tn_mpi_qu(
    qasm: str, nqubits, initial_state, mps_opts, backend="numpy"
):
    """Evaluate circuit in QASM format with Quimb using multi node multi cpu.

    Args:
        qasm (str): QASM program.
        nqubits (int): Number of qubits in the circuit
        initial_state (list): Initial state in the dense vector form. If ``None`` the default ``|00...0>`` state is used.
        mps_opts (dict): Parameters to tune the gate_opts for mps settings in ``class quimb.tensor.circuit.CircuitMPS``.
        backend (str):  Backend to perform the contraction with, e.g. ``numpy``, ``cupy``, ``jax``. Passed to ``opt_einsum``.

    Returns:
        list: Amplitudes of final state after the simulation of the circuit.
    """
    import cotengra as ctg
    from mpi4py import MPI
    from mpi4py.futures import MPICommExecutor

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    target_size = int(2**nqubits / comm.size)
    amplitudes = []

    with MPICommExecutor() as pool:
        # only need to make calls from the root process
        if pool is not None:

            if initial_state is not None:
                initial_state = init_state_tn(nqubits, initial_state)

            circ_cls = qtn.circuit.CircuitMPS if mps_opts else qtn.circuit.Circuit
            circ_quimb = circ_cls.from_openqasm2_str(
                qasm, psi0=initial_state, gate_opts=mps_opts
            )

            # options to perform the slicing and finding contraction path usign Cotengra

            opt = ctg.ReusableHyperOptimizer(
                parallel=pool,
                # make sure we generate at least 1 slice per process
                slicing_reconf_opts={"target_size": target_size},
                # uses basic greedy search algorithm to find optimal contraction path
                methods=["greedy"],
                # terminate search if contraction is cheap
                max_time="rate:1e6",
                # just uniformly sample the space
                optlib="random",
                # maximum number of trial contraction trees to generate
                max_repeats=128,
                # show the live progress of the best contraction found so far
                progbar=False,
            )
            # run the optimizer and extract the contraction tree
            interim = circ_quimb.to_dense(
                simplify_sequence="DRC", optimize=opt, rehearse=True
            )
            tree = interim["tree"]
            tensor_network = interim["tn"]
            arrays = [t.data for t in tensor_network]

            # ------------- STAGE 2: perform contraction on workers ------------- #

            # root process just submits and gather results - workers contract
            # submit contractions eagerly
            fa = [
                pool.submit(tree.contract_slice, arrays, i) for i in range(tree.nslices)
            ]

            # gather results lazily (i.e. using generator)
            amplitudes = [(c.result()).flatten() for c in fa]

    return np.array(amplitudes), rank

import cupy as cp
import numpy as np


class QiboCircuitToEinsum:
    """Convert a circuit to a Tensor Network (TN) representation.
    The circuit is first processed to an intermediate form by grouping each gate
    matrix with its corresponding qubit it is acting on to a list. It is then
    converted it to an equivalent TN expression through the class function
    state_vector_operands() following the Einstein summation convention in the
    interleave format.

    See document for detail of the format: https://docs.nvidia.com/cuda/cuquantum/python/api/generated/cuquantum.contract.html

    The output is to be used by cuQuantum's contract() for computation of the
    state vectors of the circuit.
    """

    def __init__(self, circuit, dtype="complex128"):
        self.backend = cp
        self.dtype = getattr(self.backend, dtype)
        self.init_basis_map(self.backend, dtype)
        self.init_intermediate_circuit(circuit)

    def state_vector_operands(self):
        input_bitstring = "0" * len(self.active_qubits)

        input_operands = self._get_bitstring_tensors(input_bitstring)

        (
            mode_labels,
            qubits_frontier,
            next_frontier,
        ) = self._init_mode_labels_from_qubits(self.active_qubits)

        gate_mode_labels, gate_operands = self._parse_gates_to_mode_labels_operands(
            self.gate_tensors, qubits_frontier, next_frontier
        )

        operands = input_operands + gate_operands
        mode_labels += gate_mode_labels

        out_list = []
        for key in qubits_frontier:
            out_list.append(qubits_frontier[key])

        operand_exp_interleave = [x for y in zip(
            operands, mode_labels) for x in y]
        operand_exp_interleave.append(out_list)
        return operand_exp_interleave

    def _init_mode_labels_from_qubits(self, qubits):
        n = len(qubits)
        frontier_dict = {q: i for i, q in enumerate(qubits)}
        mode_labels = [[i] for i in range(n)]
        return mode_labels, frontier_dict, n

    def _get_bitstring_tensors(self, bitstring):
        return [self.basis_map[ibit] for ibit in bitstring]

    def _parse_gates_to_mode_labels_operands(
        self, gates, qubits_frontier, next_frontier
    ):
        mode_labels = []
        operands = []

        for tensor, gate_qubits in gates:
            operands.append(tensor)
            input_mode_labels = []
            output_mode_labels = []
            for q in gate_qubits:
                input_mode_labels.append(qubits_frontier[q])
                output_mode_labels.append(next_frontier)
                qubits_frontier[q] = next_frontier
                next_frontier += 1
            mode_labels.append(output_mode_labels + input_mode_labels)
        return mode_labels, operands

    def op_shape_from_qubits(self, nqubits):
        """Modify tensor to cuQuantum shape
        (qubit_states,input_output) * qubits_involved
        """
        return (2, 2) * nqubits

    def init_intermediate_circuit(self, circuit):
        self.gate_tensors = []
        gates_qubits = []

        for gate in circuit.queue:
            gate_qubits = gate.control_qubits + gate.target_qubits
            gates_qubits.extend(gate_qubits)

            # self.gate_tensors is to extract into a list the gate matrix together with the qubit id that it is acting on
            # https://github.com/NVIDIA/cuQuantum/blob/6b6339358f859ea930907b79854b90b2db71ab92/python/cuquantum/cutensornet/_internal/circuit_parser_utils_cirq.py#L32
            required_shape = self.op_shape_from_qubits(len(gate_qubits))
            self.gate_tensors.append(
                (
                    cp.asarray(gate.matrix(), dtype=self.dtype).reshape(
                        required_shape),
                    gate_qubits,
                )
            )

        # self.active_qubits is to identify qubits with at least 1 gate acting on it in the whole circuit.
        self.active_qubits = np.unique(gates_qubits)

    def init_basis_map(self, backend, dtype):
        asarray = backend.asarray
        state_0 = asarray([1, 0], dtype=dtype)
        state_1 = asarray([0, 1], dtype=dtype)

        self.basis_map = {"0": state_0, "1": state_1}

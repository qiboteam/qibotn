import cupy as cp
import numpy as np


class QiboCircuitToEinsum:
    """This class takes a quantum circuit defined in Qibo (i.e. a Circuit object)
    and convert it to an equivalent Tensor Network (TN) representation that is formatted for
    cuQuantum' contract method to compute the state vectors.
    See document for detail: https://docs.nvidia.com/cuda/cuquantum/python/api/generated/cuquantum.contract.html

    When the class is constructed, it first process the circuit to an intermediate form by extracting the gate matrix
    and grouping each gate with its corresponding qubit it is acting on to a list.

    It is then converted it to an equivalent TN expression following the Einstein
    summation convention through the class function state_vector_operand().
    The output is to be used by cuQuantum's contract() for computation of the state vectors of the circuit.
    """

    def __init__(self, circuit, dtype="complex128"):
        def op_shape_from_qubits(nqubits):
            """This function is to modify the shape of the tensor to the required format by cuQuantum
            (qubit_states,) * input_output * qubits_involved
            """
            return (2,) * 2 * nqubits

        self.backend = cp
        self.dtype = getattr(self.backend, dtype)

        self.gate_tensors = []
        gates_qubits = []

        for gate in circuit.queue:
            gate_qubits = gate.control_qubits + gate.target_qubits
            gates_qubits.extend(gate_qubits)

            # self.gate_tensors is to extract into a list the gate matrix together with the qubit id that it is acting on
            # https://github.com/NVIDIA/cuQuantum/blob/6b6339358f859ea930907b79854b90b2db71ab92/python/cuquantum/cutensornet/_internal/circuit_parser_utils_cirq.py#L32
            required_shape = op_shape_from_qubits(len(gate_qubits))
            self.gate_tensors.append(
                (
                    cp.asarray(gate.matrix).reshape(required_shape),
                    gate_qubits,
                )
            )

        # self.active_qubits is to identify qubits with at least 1 gate acting on it in the whole circuit.
        self.active_qubits = np.unique(gates_qubits)

    def state_vector_operands(self):
        input_tensor_count = len(self.active_qubits)

        input_operands = self._get_bitstring_tensors(
            "0" * input_tensor_count, self.dtype, backend=self.backend
        )

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
        frontier_dict = {}
        n = len(qubits)
        for x in range(n):
            frontier_dict[qubits[x]] = x
        return [[i] for i in range(n)], frontier_dict, n

    def _get_bitstring_tensors(self, bitstring, dtype=np.complex128, backend=cp):
        asarray = backend.asarray
        state_0 = asarray([1, 0], dtype=dtype)
        state_1 = asarray([0, 1], dtype=dtype)

        basis_map = {"0": state_0, "1": state_1}

        operands = [basis_map[ibit] for ibit in bitstring]
        return operands

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

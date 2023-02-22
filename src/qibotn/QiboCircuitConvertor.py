import cupy as cp
import numpy as np


class QiboCircuitToEinsum:
    def __init__(self, circuit, dtype="complex128"):
        self.backend = cp
        self.dtype = getattr(self.backend, dtype)

        self.input_tensor_counter = np.zeros((circuit.nqubits,))
        self.gate_tensors = []
        for gate in circuit.queue:
            for target in gate.target_qubits:
                self.input_tensor_counter[target] += 1
            for control in gate.control_qubits:
                self.input_tensor_counter[control] += 1
            gate_qubits = gate.control_qubits + gate.target_qubits
            # self.gate_tensors is to extract into a list the gate matrix together with the qubit id that it is acting on
            self.gate_tensors.append(
                (
                    cp.asarray(gate.matrix).reshape((2,) * 2 * len(gate_qubits)),
                    gate_qubits,
                )
            )
        # self.active_qubits is to identify qubits with at least 1 gate acting on it in the whole circuit.
        self.active_qubits = [
            indx for indx, value in enumerate(self.input_tensor_counter) if value > 0
        ]

    def state_vector(self):
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

        operand_exp_interleave = [x for y in zip(operands, mode_labels) for x in y]
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

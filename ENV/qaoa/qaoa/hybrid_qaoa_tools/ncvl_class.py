from typing import List


class NonConstraintViolatingLine:

    def __init__(self, qubits: List[int],
                 rotation_qubit: int = None,
                 line_type: int = None):
        """

        :param qubits: The qubit indices the NCVL shall consist of
        :param rotation_qubit: The qubit index on which the Rz rotation for ground-state
            preparation shall be performed
        :param line_type: The type of the line (hierarchy when preparing the ground state). A value
            of 0 corresponds to an A-line, 1 to a B-line etc. (cf. notes by Anette)
        """

        self.qubits = qubits

        if rotation_qubit is None:
            self.rotation_qubit = self.qubits[0]
        elif rotation_qubit in self.qubits:
            self.rotation_qubit = rotation_qubit
        else:
            raise ValueError("rotation_qubit must be an element of qubits!")
        self.line_type = line_type

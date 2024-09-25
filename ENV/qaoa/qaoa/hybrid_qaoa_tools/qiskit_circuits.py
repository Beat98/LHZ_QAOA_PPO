from typing import List, Dict

from qiskit import QuantumCircuit
from qaoa.operator_dicts.qiskit_gates import local_field_circuit, constraint_circuit
from qaoa.operator_dicts.qiskit_logical_drivers import single_logical_circuit,\
    logical_driver_subcircuit
import numpy as np

from ncvl_class import NonConstraintViolatingLine


def get_initialization_circuit(n_qubits: int,
                               ncvls: List[NonConstraintViolatingLine]) -> QuantumCircuit:

    circuit = QuantumCircuit(n_qubits, name='init')

    current_line_type = max([ncvl.line_type for ncvl in ncvls])
    while current_line_type > 0:
        current_lines = [ncvl for ncvl in ncvls if ncvl.line_type == current_line_type]
        for line in current_lines:
            x_circuit, parameter = single_logical_circuit(len(line.qubits))
            circuit.append(x_circuit.bind_parameters({parameter: np.pi/2}).to_instruction(),
                           qargs=line.qubits)
            circuit.rz(phi=np.pi/2, qubit=line.rotation_qubit)
        current_line_type -= 1

    return circuit


def get_hybrid_subcircuits(n_qubits: int, ncvls: List[NonConstraintViolatingLine],
                           jij: List[float], constraints: List) -> Dict[str, QuantumCircuit]:
    return {
        'I': get_initialization_circuit(n_qubits, ncvls),
        'X': logical_driver_subcircuit(n_qubits, [ncvl.qubits for ncvl in ncvls]),
        'Z': local_field_circuit(jij),
        'C': constraint_circuit(n_qubits, constraints)
    }


if __name__ == '__main__':
    from qiskit import execute, Aer
    from qutip import Qobj, expect
    from full_lhz_ncvls import construct_driver_hamiltonian, get_lhz_ncvls

    n_l = 5
    n_p = n_l * (n_l - 1) // 2
    ncvls, constraints = get_lhz_ncvls(n_l, manual_constraint_indices=[0, 3, 5])
    circ = get_initialization_circuit(n_p, ncvls)
    backend = Aer.get_backend('statevector_simulator')
    output = execute(circ, backend=backend).result()

    state = output.get_statevector()
    qutip_state = Qobj(state, dims=[[2]*n_p, [1]*n_p])
    hx = construct_driver_hamiltonian(n_l, ncvls, convention='qiskit')
    energy = expect(hx, qutip_state)
    print(energy, len(ncvls))

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qaoa.operator_dicts.qiskit_gates import local_field_circuit
import math

"""
Set up the Hadamard-Gate and the CNOT-gate in terms of the R and CZ gates. This is
necessary for beeing able to set up circuits compatible with the hardware architecture by
ColdQuanta (CQ)
"""


def cq_h(c, q):
    c.r(qubit=q, phi=math.pi / 2, theta=math.pi / 2)
    c.r(qubit=q, phi=0.0, theta=math.pi)


def cq_cnot(c, q0, q1):
    cq_h(c, q1)
    c.cz(q0, q1)
    cq_h(c, q1)


def cq_single_constraint_circuit(n_qubits):
    c = Parameter('c')
    co_circ = QuantumCircuit(n_qubits, name='co')
    qubits = list(range(n_qubits))

    chain = QuantumCircuit(n_qubits)
    for i in range(n_qubits - 1):
        cq_cnot(chain, i, i + 1)

    co_circ.append(chain.to_instruction(), range(n_qubits))
    co_circ.rz(phi=c, qubit=qubits[-1])
    co_circ.append(chain.inverse().to_instruction(), range(n_qubits))

    return co_circ, c


def cq_constraint_circuit(n_qubits, constraints, parameter_name='pname'):
    circ = QuantumCircuit(n_qubits)
    parameter = Parameter(parameter_name)

    for j, cons in enumerate(constraints):
        temp_co, param = cq_single_constraint_circuit(len(cons))
        circ.append(temp_co.to_instruction({param: parameter}), cons)

    circ.name = 'constraints'
    return circ


def cq_single_rotation_circuit(n_qubits, direction='X'):
    circ = QuantumCircuit(n_qubits)
    parameter = Parameter('p' + direction)

    if direction == 'Z':
        circ.rz(parameter, qubit=range(n_qubits))
    elif direction == 'X':
        circ.r(parameter, phi=0, qubit=range(n_qubits))

    circ.name = 'rot' + direction
    return circ


def cq_interaction_circuit(n_qubits, jij, interactions):
    circ = QuantumCircuit(n_qubits)
    param = Parameter('pP')
    assert len(jij) == len(interactions), "Number of couplings does not coincide with length of jij!"
    for i, int in enumerate(interactions):
        # consider higher order interactions
        for j in range(len(int) - 1):
            cq_cnot(circ, int[j], int[j + 1])
        circ.rz(jij[i] * param, qubit=int[-1])
        for j in range(len(int) - 1)[::-1]:
            cq_cnot(circ, int[j], int[j + 1])

    circ.name = 'interactions'
    return circ


def cq_state_init(n_qubits):
    circuit = QuantumCircuit(n_qubits)
    cq_h(circuit, range(circuit.num_qubits))
    return circuit


def cq_lhz_circuits(n_qubits, local_fields, constraints):
    return {'X': cq_single_rotation_circuit(n_qubits, 'X'),
            'Z': local_field_circuit(local_fields),
            'C': cq_constraint_circuit(n_qubits, constraints),
            'I': cq_state_init(n_qubits)}


def cq_gatemodel_circuits(n_qubits, interaction_strengths, couplings):
    return{'X': cq_single_rotation_circuit(n_qubits, 'X'),
           'P': cq_interaction_circuit(n_qubits, interaction_strengths, couplings),
           'I': cq_state_init(n_qubits)}


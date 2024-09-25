from cirq import *
from sympy import Symbol
import numpy as np


def single_constraint_circuit(n_qubits, qubits):

    c = Symbol('c')
    circ = Circuit()
    circ_qubits = qubits #[LineQubit(i) for i in range(n_qubits)]

    chain = Circuit()

    for i in range(n_qubits-1):
        chain.append(CNOT(circ_qubits[i], circ_qubits[i+1]))

    circ.append(chain)
    circ.append(rz(c)(circ_qubits[-1]))
    circ.append(inverse(chain))

    return circ, c


def constraint_circuit(n_qubits, constraints, parameter_name='pname'):

    circ = Circuit()
    circ_qubits = [LineQubit(i) for i in range(n_qubits)]
    parameter = Symbol(parameter_name)
    for i, cons in enumerate(constraints):
        temp_co, param = single_constraint_circuit(len(cons), [circ_qubits[q] for q in cons])
        circ.append((resolve_parameters(temp_co, {param: parameter})), [circ_qubits[i] for i in cons])

    return circ


def single_rotation_circuit(n_qubits, direction='X'):
    circ = Circuit()
    circ_qubits = [LineQubit(i) for i in range(n_qubits)]

    parameter = Symbol('p' + direction)
    if direction == 'X':
        for qubit in circ_qubits:
            circ.append(rx(parameter)(qubit))
    elif direction == 'Z':
        for qubit in circ_qubits:
            circ.append(rz(parameter)(qubit))
    else:
        print('ERROR: Invalid direction for Rotation-Gate')
        return 0
    # circ.append(measure(*circ_qubits, key=str('result')))
    return circ


def local_field_circuit(jij):

    param = Symbol('pZ')

    circ = Circuit()
    circ_qubits = [LineQubit(i) for i in range(len(jij))]

    for i, field in enumerate(jij):
        circ.append(rz(param*field)(circ_qubits[i]))
    # circ.append(measure(*circ_qubits, key=str('result')))

    return circ


def standard_lhz_circuits(n_qubits, local_fields, constraints):
    return {'X': single_rotation_circuit(n_qubits, 'X'),
            'Z': local_field_circuit(local_fields),
            'C': constraint_circuit(n_qubits, constraints)}


if __name__ == '__main__':

    jij=[1,2,3]
    params=[1,1,1]

    loc_circ = local_field_circuit(jij)
    single_rot_circ = single_rotation_circuit(2, 'X')
    # single_const_circ, c = cq_single_constraint_circuit(5)
    const_circ = constraint_circuit(4, [[0,2,1, 3], [0,2,3]])
    print(single_rot_circ)

    s = Simulator()
    res = s.run(single_rot_circ, {'pX':np.pi/2})
    print(res)
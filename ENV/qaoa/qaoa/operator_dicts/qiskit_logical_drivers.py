from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def single_logical_circuit(n_qubits, mode='z'):
    """
    Returns circuit equivalent to exp(-i a X X X X)
    :param n_qubits: Number of qubits in single logical circuit
    :param mode: Standard mode uses combination of hadamard and z-rotation, else uses x-rotation
    :return:
    """
    parameter_l = Parameter('l')

    single_logical_driver = QuantumCircuit(n_qubits, name='ld')

    if n_qubits > 1:
        chain = QuantumCircuit(n_qubits)
        for i in range(n_qubits - 1):
            chain.cx(i + 1, i)

        single_logical_driver.append(chain.to_instruction(), range(n_qubits))

    if mode == 'z':
        single_logical_driver.h(qubit=n_qubits - 1)
        single_logical_driver.rz(phi=parameter_l, qubit=n_qubits - 1)
        single_logical_driver.h(qubit=n_qubits - 1)
    else:
        single_logical_driver.rx(theta=parameter_l, qubit=n_qubits - 1)

    if n_qubits > 1:
        single_logical_driver.append(chain.inverse().to_instruction(), range(n_qubits))

    return single_logical_driver, parameter_l


def logical_driver_subcircuit(n_qubits, logical_qubit_indizes, parameter_name='ld_p'):
    ld_circ = QuantumCircuit(n_qubits, name='log_drive')
    parameter = Parameter(parameter_name)

    for logical_qubit in logical_qubit_indizes:
        temp_circ, temp_param = single_logical_circuit(len(logical_qubit))
        ld_circ.append(temp_circ.to_instruction({temp_param: parameter}), logical_qubit)

    return ld_circ

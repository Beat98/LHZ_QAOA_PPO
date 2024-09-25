from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit


def swap_qubits_in_str(state_str, mapping):
    """
    Swaps the qubits in a state string, e. g. '010101' according to a given mapping. Note that
    strings are read from back to front due to Qiskit convention
    :param state_str: a string of a qubit configuration
    :param mapping: a mapping dictionary such as returned from get_qubit_mapping()
    :return: The string with the swapped qubits.
    """
    str_list = []
    str_list[:0] = state_str[::-1]

    new_str = str_list.copy()
    for key in mapping.keys():
        new_str[mapping[key]] = str_list[key]

    return ''.join(new_str[::-1])


def swap_qubits_in_state(state, mapping_or_circuit):
    """
    Updates the state strings in a superposition of states (qiskit statevector of counts)
    :param state: A 'state' as a dictionary of counts, e. g. {'00011': 32, "01010':25}
    :param mapping_or_circuit: a mapping such as generated form get_qubit_mapping() or a circuit. If
    a circuit is given, get_qubit_mapping() will be called for that circuit.
    :return: The state with the updated strings.
    """
    if type(mapping_or_circuit) is QuantumCircuit:
        mapping = get_qubit_mapping(mapping_or_circuit)
    elif type(mapping_or_circuit) is dict:
        mapping = mapping_or_circuit
    else:
        raise TypeError("Either mapping or circuit must be given.")

    new_state = dict()
    # print(state.keys())
    for state_str in state.keys():
        new_str = swap_qubits_in_str(state_str, mapping)
        new_state[new_str] = state[state_str]

    return new_state


def get_swap_operations(circ):
    """
    Returns a list of SWAP-operations in a qiskit-circuit. If there is e. g. a SWAP gate between
    qubits 0 and 2 and between 1 and 3, the output would be [[0, 2], [1, 3]].
    :param circ: The circuit in which the desired SWAP operations are implemented
    :return: A list of lists. In the sublists, the qubit indices being swapped, are listed.
    """
    dag = circuit_to_dag(circ)
    swap_dict = []
    for node in dag.topological_op_nodes():
        if node.name == 'swap':
            swap_dict += [[node.qargs[0].index, node.qargs[1].index]]

    return swap_dict


def get_qubit_mapping(circ):
    """
    Returns a dictionary that maps the qubit's indices before performing a circuit to the
    indices afterwards, as these can differ due to SWAP-gates in the circuit. If, for example,
    there are SWAP operations between qubits 0, 1 and 0, 2, the output will be
    {0: 2, 1: 0, 2: 1}. Note that only qubits that have changed the label are listed in the dictionary.
    :param circ: The circuit that contains the SWAP gates.
    :return: dictionary with the mapping.
    """
    swap_operations = get_swap_operations(circ)
    q_list = list(set([item for swap in swap_operations for item in swap]))
    q_list_new = q_list.copy()

    for swap in swap_operations:
        temp = q_list_new[q_list.index(swap[0])]
        q_list_new[q_list.index(swap[0])] = q_list_new[q_list.index(swap[1])]
        q_list_new[q_list.index(swap[1])] = temp

    mapping = dict(zip(q_list, q_list_new))

    for key, val in list(mapping.items()):
        if val == key:
            del mapping[key]

    return mapping

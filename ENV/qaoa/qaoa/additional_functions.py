from itertools import product

import numpy as np
from lhz.core import n_logical, qubit_dict
from qutip import Qobj

def decompose_state(state: Qobj, Np: int):
    decomposition_dict = {}

    for i, amplitude in enumerate(state.data.toarray()):
        probability = np.abs(amplitude) ** 2
        qubit_config = tuple(map(int, format(i, '0' + str(Np) + 'b')))
        decomposition_dict[qubit_config] = probability[0]

    return decomposition_dict

# def get_configs(Np: int):
#
#     config_list = np.zeros((2**Np, Np))
#     for i in range(2**Np):
#         config_list[i,:] = tuple(map(int, format(i, '0' + str(Np) + 'b')))
#
#     return config_list

def get_configs(Np):
    configs = list(product([1, -1], repeat=Np))
    return np.array(configs)


def get_spanning_tree_configs(conf, num_spanningtrees):
    Np = len(conf)
    Nl = n_logical(Np)

    qd = qubit_dict(Nl, "standard")

    logical_conf_list = np.ones((num_spanningtrees, Nl))

    for spanning_tree_idx in range(num_spanningtrees):
        for i in range(Nl):
            if spanning_tree_idx != i:
                sorted_tuple = tuple(sorted((spanning_tree_idx, i)))
                # logical_conf_list[spanning_tree_idx, i] = -1 if conf[qd[sorted_tuple]] == 1 else 1
                logical_conf_list[spanning_tree_idx, i] = conf[qd[sorted_tuple]]

    return logical_conf_list


def logical_energy(log_conf, Jij):
    Nl = len(log_conf)
    i_upper, j_upper = np.triu_indices(Nl, k=1)
    logical_energy = np.einsum('i,i,i', log_conf[i_upper], Jij, log_conf[j_upper])
    return logical_energy






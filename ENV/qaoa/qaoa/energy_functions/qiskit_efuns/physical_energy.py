"""
Created on Wed Jan 12 16:28 2022

@author: kili, michi
"""
from typing import Callable, List

import numpy as np
from functools import partial


def energy_of_physical_configuration(jij, constraints, costr, bin_conf):
    # bin_conf[::-1] because in qiskit the qubits are labeled like q_3 q_2 q_1 q_0
    conf = [1 if b == '0' or b == 0 else -1 for b in bin_conf[::-1]]
    e = np.sum(np.array(jij) * np.array(conf))

    if type(costr) is float or type(costr) is int:
        costr_list = [costr] * len(constraints)

    else:
        assert len(constraints) == len(costr), "length of constraints must match length of costr!"
        costr_list = costr

    for i, c in enumerate(constraints):
        e += -costr_list[i] * (np.prod(np.array(conf)[c]) - 1) / 2
        # e += -costr_list[i] * (np.prod(np.array(conf)[c]))
    return e


def create_efun_physical(jij: List[float], constraints: List[List[int]], costr) -> Callable[[str], float]:
    """
    Creates energy function which calculates energy of a
    physical configuration (in qiskit bitstring format)
    Attention: only references of local fields and constraints are saved!
    :param jij: Local fields
    :param constraints: List of lists of indizes for constraint terms
    :param costr: Either single value or list of values,
    containing energy penalties for violating constraints
    :return: Returns a function taking a binary configuration as input and evaluating the energy
    """
    return partial(energy_of_physical_configuration, jij, constraints, costr)

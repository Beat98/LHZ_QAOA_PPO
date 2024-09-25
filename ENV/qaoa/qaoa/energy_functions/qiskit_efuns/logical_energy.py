"""
Created on Wed Jan 12 16:28 2022

@author: kili, michi
"""
from functools import partial
from typing import Callable, List


def energy_of_logical_configuration(logical_graph, jij, bin_conf):
    conf = [1 if b == '0' or b == 0 else -1 for b in bin_conf[::-1]]
    # conf = [-1 if b == '0' or b == 0 else 1 for b in bin_conf]
    # print(conf)
    e = 0
    # TODO: Implement for higher order interactions
    for i, co in enumerate(logical_graph):
        e += jij[i] * conf[co[0]] * conf[co[1]]
    # print(e)
    return e


def create_efun_logical(logical_graph: List[List[int]], jij: List[float]) -> Callable[[str], float]:
    """
    Creates energy function which calculates energy of a
    logical configuration (in qiskit bitstring format) for a graph with two-body interactions
    Attention: only references of local fields and the logical graph are saved!
    :param logical_graph: Logical two-body graph
    :param jij: Interaction strengths
    :return: Function calculating energy of a logical configuration
    """
    return partial(energy_of_logical_configuration, logical_graph, jij)


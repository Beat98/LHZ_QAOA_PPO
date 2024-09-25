"""
Created on Wed Jan 12 16:28 2022

@author: anita
"""
from functools import partial
from typing import Callable
# from roelands_super_code import spanning_tree_things


# TODO for Anita
# for Kilian: if spanning_trees is a reference, does the returned function have the reference or a copy?
# answer: it is a reference! careful
def create_efun_logical_decoding(jij, spanning_trees, *args) -> Callable[[str], float]:

    def my_cool_function(inpu: str):
        for sp in spanning_trees:
            pass

        return 1.0

    return my_cool_function


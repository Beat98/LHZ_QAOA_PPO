"""
Created on Fri Apr  6 13:18:46 2018

@author: kili

Basic functions for working in the LHZ architecture.

"""

import numpy as np

n_physical = lambda n_l: int(n_l * (n_l - 1) / 2)
n_physical_finiterange = lambda n_l, r: r * n_l - r * (r + 1) // 2
n_logical = lambda n_p: int(1 / 2 + np.sqrt(1 + 8 * n_p) / 2)
n_logical_finiterange = lambda n_p, r: n_p // r + (r + 1) // 2


def create_constraintlist(n_l):
    constraints = []

    inds = np.triu_indices(n_l - 1, 1)

    for p1, p2 in zip(inds[0], inds[1]):
        tempc = [(p1, p2), (p1, p2 + 1), (p1 + 1, p2 + 1)]
        if p2 != p1 + 1:
            tempc.append((p1 + 1, p2))
        constraints.append(tempc)

    return constraints


def create_constraintlist_finiterange(n_l, r):
    constraints = []

    inds = np.zeros((n_l - 1, n_l - 1))
    inds[np.triu_indices_from(inds, 1)] = 1
    inds[np.triu_indices_from(inds, r - 1 + 1)] = 0

    inds = np.where(inds)

    for p1, p2 in zip(inds[0], inds[1]):
        tempc = [(p1, p2), (p1, p2 + 1), (p1 + 1, p2 + 1)]
        if p2 != p1 + 1:
            tempc.append((p1 + 1, p2))
        constraints.append(tempc)

    assert len(constraints) == int((r - 1) * n_l - r * (r + 1) / 2 + 1), 'test'

    return constraints


def qubit_dict(n_l, case='standard', r=None):
    qdict = {}

    if case == 'standard':
        inds = np.triu_indices(n_l, 1)
        for i, (p1, p2) in enumerate(zip(inds[0], inds[1])):
            qdict[(p1, p2)] = i
    elif case == 'bottom_left_to_right':
        c = 0
        for j in range(1, n_l):
            for i in range(n_l - j):
                qdict[(i, j + i)] = c
                c += 1
    elif case == 'finite-range':
        assert r is not None, 'r must be set'
        assert r < n_l, 'r must be smaller than Nlogical'

        inds = np.zeros((n_l, n_l))
        inds[np.triu_indices_from(inds, 1)] = 1
        inds[np.triu_indices_from(inds, r + 1)] = 0

        inds = np.where(inds)

        for i, (p1, p2) in enumerate(zip(inds[0], inds[1])):
            qdict[(p1, p2)] = i

    else:
        raise ValueError('case not defined')

    return qdict


def standard_constraints(nl):
    qd = qubit_dict(nl)
    cl = create_constraintlist(nl)
    return [[qd[c] for c in cs] for cs in cl]

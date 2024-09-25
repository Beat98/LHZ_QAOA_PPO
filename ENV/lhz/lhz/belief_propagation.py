#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:20:42 2018

@author: kili
"""

# % ERROR CORRECTION FOR QUANTUM ANNEALING ARCHITECTURE
# % requires: nothing
# % author: Fernando Pastawski
# % license: GPL3
# % If this code is usefull to you, please cite the accompanying article


import numpy as np
n_logical = lambda n_p: int(1 / 2 + np.sqrt(1 + 8 * n_p) / 2)

eps = np.finfo(np.float).eps


def BPdecoder(bitstr, probs, nits=4):
    """
    decoding of a bitstring 'bitstr' which is wrong with probability 'probs'
    in standard LHZ architecture
    qubits sorted as in lhz.core.qubit_dict with qubit_dict(Nl, case='standard')
        format of bitstr: [12 13 14 ... 1N 23 24 ... 2N 34 ... 3N .... N-1 N]
        
    bitstr.. bitstring to be decoded
    probs..  probabilities of readouts in bitstring being wrong
    nits..   how many iterations of the belief propagation should be done
    """

    assert len(bitstr) == len(probs), 'bitstr and probs must be of same length'

    N = n_logical(len(bitstr))
    configuration = np.zeros((N, N))
    configuration[np.triu_indices(N, 1)] = bitstr

    p_errror = np.zeros((N, N))
    p_errror[np.triu_indices(N, 1)] = probs
    p_errror = p_errror + p_errror.transpose()

    for i in range(N):
        configuration[i, i] = 1
        for j in range(i + 1, N):
            configuration[j, i] = configuration[i, j]

    for it in range(nits):  # Number of iterations;
        Pm1 = (p_errror - 1 / 2) * configuration + 1 / 2  # Likelihood of being -1  ( per or 1-per )
        Pp1 = (configuration + 1) / 2 - p_errror * configuration  # Likelihood of being +1  ( per or 1-per )
        for i in range(N):
            for j in range(i + 1, N):  # j > i
                for k in [l for l in range(N) if l != i and l != j]:
                    # Calculate the likelihood (unnormalized) of (i,j) being pm1
                    # given neighborhood and associated trust value
                    Pm1[i, j] = Pm1[i, j] \
                                * ((configuration[i, k] * configuration[k, j] == -1) * (
                                               (1 - p_errror[i, k]) * (1 - p_errror[k, j]) + p_errror[i, k] * p_errror[k, j])
                                   + (configuration[i, k] * configuration[k, j] == 1) * (
                                               (1 - p_errror[i, k]) * p_errror[k, j] + p_errror[i, k] * (1 - p_errror[k, j])))
                    Pp1[i, j] = Pp1[i, j] \
                                * ((configuration[i, k] * configuration[k, j] == 1) * (
                                               (1 - p_errror[i, k]) * (1 - p_errror[k, j]) + p_errror[i, k] * p_errror[k, j])
                                   + (configuration[i, k] * configuration[k, j] == -1) * (
                                               (1 - p_errror[i, k]) * p_errror[k, j] + p_errror[i, k] * (1 - p_errror[k, j])))
        # Update our current best guess in mu and 
        # calculate normalized marginal distributions
        for i in range(N):
            for j in range(i + 1, N):
                configuration[i, j] = 2 * (Pp1[i, j] >= Pm1[i, j]) - 1  # Recalculate our favorite value
                configuration[j, i] = configuration[i, j]
                # Recalculate relative probabilities use eps to
                # regulate overconfidence
                p_errror[i, j] = max(min(Pm1[i, j], Pp1[i, j]) / (Pp1[i, j] + Pm1[i, j]), eps)
                p_errror[j, i] = p_errror[i, j]

    #        print('mu it:\n', mu)
    #        print('per it:\n', per)

    return configuration[np.triu_indices(N, 1)], p_errror[np.triu_indices(N, 1)]


if __name__ == '__main__':
    p = 0.06

    #######################
    #
    #   change input bitstring here, press play
    #
    #######################

    # a correct one:
    bitstr = [-1, -1, -1, 1, 1, 1, -1, 1, -1, -1]
    per = p * np.ones_like(bitstr)

    # introduce errors:
    #    bitstr = [-1,-1,-1,1,1,1,-1,-1,1,1] # this one is weird, in LHZ it shouldn't be correct (3 body constraint flipped)

    #    bitstr = [-1,-1,-1,1,1,1,-1,1,-1,-1]

    flipinds = [7, 6]
    for fi in flipinds:
        bitstr[fi] *= -1

    print('starting bitstr:')
    print(np.array(bitstr, dtype=int))

    print('---------------------')


    bitstr = [1, 1, 1, -1, 1, 1]
    per = [0.05]*6
    muret, perret = BPdecoder(bitstr, per)

    print('corrected bitstr:')
    print(np.array(muret, dtype=int))
    print('probs to be inncorrect:')
    print(np.round(perret, 3))
    # print('---------------------')
    # print('mu ret:\n', muret)
    # print('per ret:\n', perret)

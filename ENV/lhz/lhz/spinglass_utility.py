#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:00:49 2018

@author: kili
"""
import numpy as np
from lhz.core import n_logical, n_physical, n_physical_finiterange, qubit_dict, create_constraintlist
# from utility import qi_reverse_reverse
# from qaoa_class import qaoa
from qutip import tensor, basis


def gray_code_fliplist(n):
    fliplist = []
    pre = 2 ** n
    for i in range(1, 2 ** n):
        c = 2 ** n + int(i / 2) ^ i
        fliplist.append(n - len(bin(c ^ pre)) + 2)
        pre = c
    return fliplist


def all_energies(jij, return_confs=False):
    """
    jij in linear format
    """
    n_p = len(jij)
    n_l = n_logical(n_p)

    assert n_p == n_physical(n_l), 'linear format of Jij wrong'

    jij_l = np.zeros((n_l, n_l))
    jij_l[np.triu_indices_from(jij_l, 1)] = np.array(jij)
    jij_l = jij_l + jij_l.transpose()

    es = []
    conf = [-1] * n_l

    ee = np.sum(np.multiply(
        np.ufunc.reduce(np.multiply, np.ix_(*([conf] * 2))), jij_l)) / 2

    es.append(ee)

    if return_confs:
        confs = []
        confs.append(conf.copy())

    pre = 2 ** n_l
    for i in range(1, 2 ** n_l):
        c = 2 ** n_l + int(i / 2) ^ i
        iFlip = n_l - len(bin(c ^ pre)) + 2
        conf[iFlip] *= -1
        pre = c

        if return_confs:
            confs.append(conf.copy())

        ee += 2 * conf[iFlip] * np.sum(jij_l[:, iFlip] * conf)
        es.append(ee)

    if return_confs:
        return es, confs

    return es


def get_lowest_state(Jij):
    Np = len(Jij)
    Nl = n_logical(Np)

    assert Np == n_physical(Nl), 'linear format of Jij wrong'

    Jijl = np.zeros((Nl, Nl))
    Jijl[np.triu_indices_from(Jijl, 1)] = np.array(Jij)
    Jijl = Jijl + Jijl.transpose()

    minconf = [-1] * Nl
    mine = np.sum(np.multiply(np.ufunc.reduce(np.multiply, np.ix_(*([minconf] * 2))), Jijl)) / 2
    et = mine

    conf = [-1] * Nl
    pre = 2 ** Nl
    for i in range(1, 2 ** Nl):
        c = 2 ** Nl + int(i / 2) ^ i
        iFlip = Nl - len(bin(c ^ pre)) + 2
        conf[iFlip] *= -1
        pre = c

        et += 2 * conf[iFlip] * np.sum(Jijl[:, iFlip] * conf)

        if et <= mine:
            mine = et
            minconf = conf.copy()

    return mine, minconf


def get_lowest_states(jij):
    n_p = len(jij)
    n_l = n_logical(n_p)

    assert n_p == n_physical(n_l), 'linear format of Jij wrong'

    jijl = np.zeros((n_l, n_l))
    jijl[np.triu_indices_from(jijl, 1)] = np.array(jij)
    jijl = jijl + jijl.transpose()

    conf = [-1] * n_l
    mine = np.sum(np.multiply(np.ufunc.reduce(np.multiply, np.ix_(*([conf] * 2))), jijl)) / 2
    et = mine
    minconfs = [conf.copy()]

    pre = 2 ** n_l
    for i in range(1, 2 ** n_l):
        c = 2 ** n_l + int(i / 2) ^ i
        i_flip = n_l - len(bin(c ^ pre)) + 2
        conf[i_flip] *= -1
        pre = c
        et += 2 * conf[i_flip] * np.sum(jijl[:, i_flip] * conf)

        if et < mine:
            mine = et
            minconfs = [conf.copy()]
        elif abs(et-mine) < 1e-9:
            mine = et
            minconfs += [conf.copy()]

    return mine, minconfs


def translate_configuration_logical_to_physical(conf, qubit_dict_case='standard', r=None):
    Nl = len(conf)

    if r is not None:
        qd = qubit_dict(Nl, 'finite-range', r)
        Np = n_physical_finiterange(Nl, r)
    else:
        Np = n_physical(Nl)
        qd = qubit_dict(Nl, qubit_dict_case)

    physconf = [0] * Np

    for pair, i in qd.items():
        physconf[i] = conf[pair[0]] * conf[pair[1]]

    return physconf


def translate_configuration_physical_to_logical(conf, qubit_dict_case='standard', r=None):
    Np = len(conf)

    if r is not None:
        assert False, 'implement this'
    else:
        Nl = n_logical(Np)
        qd = qubit_dict(Nl, qubit_dict_case)

    logconf = [1] * Nl

    for i in range(1, Nl):
        logconf[i] = logconf[0] * conf[qd[(0, i)]]

    return logconf, [-xxx for xxx in logconf]


def create_qutip_wf(vec):
    """
    +1 for u (gs of -sigma_z) or sigma_z u = +1 u
    -1 for d (gs of sigma_z) or sigma_z d = -1 d
    + for u + d (gs of -sigma_x)
    - for u - d (gs of sigma_x)
    """

    prod = []
    for i in vec:
        if i == 1:
            prod.append(basis(2, 0))
        elif i == -1:
            prod.append(basis(2, 1))
        elif i == '+':
            prod.append((basis(2, 0) + basis(2, 1)).unit())
        elif i == '-':
            prod.append((basis(2, 0) - basis(2, 1)).unit())
        else:
            return None

    return tensor(prod)


def all_lhz_energies(Jij, constraints, cstr):
    Np = len(Jij)

    es = []
    for conf in all_confs(Np):
        e = np.sum(np.array(Jij) * np.array(conf))

        for c in constraints:
            e += -cstr * (np.prod(np.array(conf)[c]) - 1)/2

        es.append(e)
    return es


def single_lhz_energy(Jij, constraints, cstr, conf):
    e = np.sum(np.array(Jij) * np.array(conf))

    for c in constraints:
        e += -cstr * (np.prod(np.array(conf)[c]) - 1)/2

    return e


def inf_temp_hist(Jij, constraints, cstr, num_bins):
    es = all_lhz_energies(Jij, constraints, cstr)

    return np.histogram(es, num_bins)


def all_confs(N):
    return [[2 * int(j) - 1 for j in list(np.binary_repr(i, width=N))] for i in range(2 ** N)]


def get_classical_gs_energy(Jij):
    return min(all_energies(Jij, return_confs=False))


def lhz_energy_single_state(Jij, constraints, cstr, conf):
    """ qutip -> rigetti: -1 -> 1, 1 -> 0"""
    e = np.sum(np.array(Jij) * (1 - 2 * np.array(conf)))

    for c in constraints:
        t = -cstr
        #        print(c)
        for ic in c:
            t *= 1 - 2 * conf[ic]
        #        print(t)
        e += t

    return e


def num_of_violated_constraints(bit_conf, constraints=None):
    """ -1, 1 format"""

    ret = 0

    if constraints is None:
        qd = qubit_dict(n_logical(len(bit_conf)))
        cs = create_constraintlist(n_logical(len(bit_conf)))

        constraints = [[qd[cc] for cc in cxx] for cxx in cs]

    for c in constraints:
        t = 1
        for ic in c:
            t *= bit_conf[ic]
        if t == -1:
            ret += 1

    return ret


# %% testing
# if __name__ == '__main__':
#     pass
    # jij = [-1, -1, -1]
    # x = get_lowest_states(jij)
    #    from lhz.core import create_constraintlist, qubit_dict
    #    Jij = [0.80, -0.62, -0.25, 0.88, -0.87, 0.34]
    # from time import time
    #
    # Nl = 12
    # Np = n_physical(Nl)
    #
    # Jij = 2 * (np.random.random(size=Np) - 0.5)
    #
    # t0 = time()
    #    a = np.array(all_energies2(Jij))
    # t1 = time()
    # b = np.array(all_energies(Jij))
    # t2 = time()
    # es, confs = all_energies(Jij, True)
    # es = np.array(es)
    # t3 = time()
    #    d = np.array(all_energies_4(Jij))
    # t4 = time()

    #    a.sort()
    # b.sort()
    #    c.sort()
    #    d.sort()

    # t5 = time()
    # ee, ev = get_lowest_state(Jij)
    # t6 = time()
    #
    # print(t1 - t0, t2 - t1, t3 - t2, t4 - t3, t6 - t5)
#    print(sum(abs(a-b)))

#    %timeit all_energies_2(Jij)
#    
#    cs_bf = create_constraintlist(Nl)
#    qd = qubit_dict(Nl)
#    connections = [[qd[ci] for ci in c] for c in cs_bf]
#    
##    connections = [[0,1,3],[1,2,4],[1,3,4,5]]
#    
##    h = inf_temp_hist(Jij, connections, 3.0, 24)
#    es = all_lhz_energies(Jij, connections, 3.0)
#    h = np.histogram(es, 24)ls
#    import matplotlib.pyplot as plt
#    plt.plot(h[1][0:-1],h[0],'o-')
#    
#    #connections = [[0,1,2]]

# confi = [0, 1, 1, 0, 1, 0]
#    confi = [1, 0, 0, 1, 0, 1]
#    
#    print(lhz_energy_single_state(Jij, connections, 3.0, confi))
#    
##    from qaoa_class import qaoa
##    from qutip import expect
#    
#    q = qaoa(Jij, connections)
#    ees, evecs = q.HP.eigenstates()
#    gssz = expect(q.sz_list, evecs[0])
#    
#    confs = []
#    for state in evecs:
#        confs.append(np.array([(1-x)/2 for x in expect(q.sz_list, state)]))
#    
#    gs_rigetti_format = np.array([(1-x)/2 for x in gssz])
#    
#    print(lhz_energies_classical_spinglass(Jij, connections, 3.0, confs))
#    
#    print(ees, gssz)
#    print(confi)
#    print(gs_rigetti_format)

"""
Created on Fri Apr  6 13:26:13 2018

@author: kili


Functions for creating qutip-Hamiltonians


"""

from qutip import qeye, sigmax, sigmay, sigmaz, tensor, sigmam, sigmap, commutator
import numpy as np
import lhz.core as lhz_core


def comm(x, y):
    return not np.any(np.abs((x * y - y * x).data.data) > 0)


def Hdict_physical(Jij, cstrength=3.0, constraints=None):
    """
    Jij...             Jij matrix for physical spins, 
                       format:  (check conventions)
                          - either linear, length N_phsical
                              eg. [J_01, J_02, J_03, J_12, J_13, J_23]
                          - or original Jij matrix (NlogicalxNlogical matrix)
    cstrength..        prefactor for all constraints
    constraints..      n-body constraints of the form 'cstrength*prod_i sigma_i^z'
                       constraint fulfilled, 0 energy,
                       constraint violated,  cstrength energy penalty
                       in total penalty = cstrength*num_violated_constraints
    :returns           H = sum{J_i sigma_i^z} + cstrength*sum{1-prod_i sigma_i^z}
    """
    # check and convert inputs
    shape = np.shape(Jij)
    if len(shape) == 1:
        Np = len(Jij)
        Jijp = Jij

        assert Np == lhz_core.n_physical(lhz_core.n_logical(Np)), \
            'linear format of Jij wrong'

    elif len(shape) == 2:  # input as Jij matrix, test for list of lists
        Np = lhz_core.n_physical(shape[0])
        Jijp = np.array(Jij)[np.triu_indices_from(Jij, 1)]
    else:
        pass

    Nl = lhz_core.n_logical(Np)

    # prepare constraints
    if constraints is None:
        constraints = []
        cs = lhz_core.create_constraintlist(Nl)
    elif isinstance(constraints[0][0], tuple):
        cs = constraints[:]
        constraints = []

    qubit_dict = lhz_core.qubit_dict(Nl)
    if len(constraints) == 0:
        for c in cs:
            constraints.append([qubit_dict[ci] for ci in c])

    assert isinstance(constraints[0][0], int)

    # prepare paulis for physical hamiltonians
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(Np):
        op_list = [si] * Np
        op_list[n] = sx
        sx_list.append(tensor(op_list))
        op_list[n] = sy
        sy_list.append(tensor(op_list))
        op_list[n] = sz
        sz_list.append(tensor(op_list))

    # hamiltonians
    HX = 0
    HY = 0
    HYj = 0
    HZ = 0
    for i in range(Np):
        HX += sx_list[i]
        HY += sy_list[i]
        HYj += Jijp[i] * sy_list[i]
        HZ += Jijp[i] * sz_list[i]

    HCs = []
    NCs = []
    hhs = []
    for c in constraints:
        h = 1
        for ci in c:
            h *= sz_list[ci]
        # HCs.append(cstrength*(1*1-h))
        hhs.append(h)
        HCs.append(cstrength * (1 - h) / 2)
        NCs.append((1 - h) / 2)
        # h = -sz_list[c[0]]
        # for ci in range(len(c) - 1):
        #     h *= sz_list[c[ci + 1]]
        # HCs.append(cstrength * (1 + h))

    HC = sum(HCs)
    HP = HZ + HC

    # logical X-lines
    HVs = []

    for i in range(Nl):
        thv = 1
        for (k, v), ind in qubit_dict.items():
            if k == i or v == i:
                thv *= sx_list[ind]
                # print(ind)
        HVs += [thv]



    comm_HX_HZ = 1j * commutator(HX, HZ)
    comm_HX_HC = 1j * commutator(HX, HC)



    Hdict = {'X': HX, 'Y': HY, 'Z': HZ, 'C': HC, 'P': HP, 'test': hhs,
             'x': sx_list, 'y': sy_list, 'z': sz_list, 'c': HCs,
             'Nc': sum(NCs), 'U': HYj, 'Vs': HVs, 'V': sum(HVs),
             'A': comm_HX_HZ, 'B': comm_HX_HC}
    return Hdict


def hdict_physical_not_full(jij, constraints, cstrength=3.0, log_line_indizes=None):
    """
    Jij...             Jij matrix for physical spins,
    cstrength..        prefactor for all constraints
    constraints..      n-body constraints of the form 'cstrength*prod_i sigma_i^z'
                       constraint fulfilled, 0 energy,
                       constraint violated,  cstrength energy penalty
                       in total penalty = cstrength*num_violated_constraints
    :returns           H = sum{J_i sigma_i^z} + cstrength*sum{1-prod_i sigma_i^z}
    """
    # check and convert inputs
    n_p = len(jij)

    # prepare paulis for physical hamiltonians
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(n_p):
        op_list = [si] * n_p
        op_list[n] = sx
        sx_list.append(tensor(op_list))
        op_list[n] = sy
        sy_list.append(tensor(op_list))
        op_list[n] = sz
        sz_list.append(tensor(op_list))

    # hamiltonians
    HX = 0
    HY = 0
    HYj = 0
    HZ = 0
    for i in range(n_p):
        HX += sx_list[i]
        HY += sy_list[i]
        HYj += jij[i] * sy_list[i]
        HZ += jij[i] * sz_list[i]

    HCs = []
    NCs = []
    hhs = []
    for c in constraints:
        h = 1
        for ci in c:
            h *= sz_list[ci]
        # HCs.append(cstrength*(1*1-h))
        hhs.append(h)
        HCs.append(cstrength * (1 - h) / 2)
        # HCs.append(-cstrength*h)
        NCs.append((1 - h) / 2)

    HC = sum(HCs)
    HP = HZ + HC

    h_dict = {'X': HX, 'Y': HY, 'Z': HZ, 'C': HC, 'P': HP, 'test': hhs,
              'x': sx_list, 'y': sy_list, 'z': sz_list, 'c': HCs,
              'Nc': sum(NCs), 'U': HYj, 'A': 1j * commutator(HX, HZ), 'B': 1j * commutator(HX, HC),
              # 'D': commutator(HZ, commutator(HX, HC)), 'E': commutator(HX, commutator(HX, HC)),
              # 'F': commutator(HC, commutator(HX, HC)), 'G': commutator(HZ, commutator(HX, HZ)),
              'H': 1j * commutator(HX, commutator(HX, commutator(HX, HC))), 'I': 1j * commutator(HC, commutator(HC, commutator(HX, HC))),
              'J': 1j * commutator(HX, commutator(HZ, commutator(HX, HC))), 'K': 1j * commutator(HX, commutator(HC, commutator(HX, HC))),
              'L': 1j * commutator(HZ, commutator(HC, commutator(HX, HC)))}

    if log_line_indizes is not None:
        HVs = []

        for log_line in log_line_indizes:
            temp_hv = 1
            for ind in log_line:
                temp_hv *= sx_list[ind]
            HVs += [temp_hv]

        h_dict['L'] = sum(HVs)

    return h_dict


def Hdict_physical_finiterange(Jij, r=2, cstrength=3.0):  # , constraints=None):
    """
    Jij...             Jij matrix for physical spins, 
                       format:  (check conventions)
                          - either linear, length N_phsical
                              eg. [J_01, J_02, J_03, J_12, J_13, J_23]
                          - or original Jij matrix (NlogicalxNlogical matrix)
    r..                range of interactions
    cstrength..        prefactor for all constraints
    constraints..      n-body constraints of the form 'cstrength*(1-prod_i sigma_i^z)'
    """
    # check and convert inputs
    shape = np.shape(Jij)
    if len(shape) == 1:
        assert False, 'for this Jij needs to be in square format'

    elif len(shape) == 2:  # input as Jij matrix, test for list of lists
        Nl = shape[0]
        #        Np = lhz_core.Nphysical(Nl)

        inds = np.zeros((Nl, Nl))
        inds[np.triu_indices_from(inds, 1)] = 1
        inds[np.triu_indices_from(inds, r + 1)] = 0

        Jijp = np.array(Jij)[np.where(inds)]


    else:
        pass

    # prepare constraints
    constraints = []
    cs = lhz_core.create_constraintlist_finiterange(Nl, r)

    qubit_dict = lhz_core.qubit_dict(Nl, case='finite-range', r=r)
    Np = len(qubit_dict)

    for c in cs:
        constraints.append([qubit_dict[ci] for ci in c])

    if len(constraints) > 0:  # for range = 1 (useless?), no constaints
        assert isinstance(constraints[0][0], int)

    # prepare paulis for physical hamiltonians
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(Np):
        op_list = [si] * Np
        op_list[n] = sx
        sx_list.append(tensor(op_list))
        op_list[n] = sy
        sy_list.append(tensor(op_list))
        op_list[n] = sz
        sz_list.append(tensor(op_list))

    # hamiltonians
    HX = 0
    HY = 0
    HZ = 0
    for i in range(Np):
        HX += sx_list[i]
        HY += sy_list[i]
        HZ += Jijp[i] * sz_list[i]

    HCs = []
    for c in constraints:
        h = -1
        for ci in c:
            h *= sz_list[ci]
        HCs.append(cstrength * (1 + h))
    # for c in constraints:
    #     # h = cstrength
    #     # for ci in c:
    #     #     h *= sz_list[ci]
    #     #
    #     # HCs.append(1+h)
    #     h = -sz_list[c[0]]
    #     #        print(c)
    #     #        print(c[0])
    #     for ci in range(len(c) - 1):
    #         h *= sz_list[c[ci + 1]]
    #     #            print(c[ci+1])
    #
    #     HCs.append(cstrength * (1 + h))

    HC = sum(HCs)
    HP = HZ + HC

    Hdict = {'X': HX, 'Y': HY, 'Z': HZ, 'C': HC, 'P': HP,
             'x': sx_list, 'y': sy_list, 'z': sz_list, 'c': HCs}
    return Hdict


keys_logical = 'XYPUxyz'


def Hdict_logical(Jij):
    """
    Jij...             Jij matrix for logical spins,  (real)
                       format:  (check conventions)
                          - either linear, length N_phsical
                              eg. [J_01, J_02, J_03, J_12, J_13, J_23]
                          - or original Jij matrix (NlogicalxNlogical matrix)
    """

    # check and convert inputs
    shape = np.shape(Jij)
    if len(shape) == 1:
        Np = len(Jij)
        Nl = lhz_core.n_logical(Np)

        assert Np == lhz_core.n_physical(Nl), 'linear format of Jij wrong'

        Jijl = np.zeros((Nl, Nl))
        Jijl[np.triu_indices_from(Jijl, 1)] = np.array(Jij)
        Jijl = Jijl + Jijl.transpose()
    elif len(shape) == 2:  # input as Jij matrix, test for list of lists
        Nl = shape[0]
        Jijl = np.array(Jij)
    else:
        pass

    # prepare paulis for physical hamiltonians
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(Nl):
        op_list = [si] * Nl
        op_list[n] = sx
        sx_list.append(tensor(op_list))
        op_list[n] = sy
        sy_list.append(tensor(op_list))
        op_list[n] = sz
        sz_list.append(tensor(op_list))

    # hamiltonians
    HX = 0
    HY = 0
    HYj = 0
    HP = 0
    for i in range(Nl):
        HX += sx_list[i]
        HY += sy_list[i]
        for j in range(i):
            HP += Jijl[i, j] * sz_list[i] * sz_list[j]
            HYj += Jijl[i, j] * (sy_list[i] * sz_list[j] + sz_list[i] * sy_list[j])

    Hdict = {'X': HX, 'Y': HY, 'P': HP, 'x': sx_list, 'y': sy_list, 'z': sz_list, 'U': HYj}
    return Hdict


def Hdict_physical_dynfields(Jij, cstrength=3.0, constraints=None):
    shape = np.shape(Jij)
    if len(shape) == 1:
        Np = len(Jij)
        Jijp = Jij

        assert Np == lhz_core.n_physical(lhz_core.n_logical(Np)), \
            'linear format of Jij wrong'

    elif len(shape) == 2:  # input as Jij matrix, test for list of lists
        Np = lhz_core.n_physical(shape[0])
        Jijp = np.array(Jij)[np.triu_indices_from(Jij, 1)]
    else:
        pass

    Nl = lhz_core.n_logical(Np)
    Nfull = 2 * Np

    # prepare constraints
    if constraints is None:
        constraints = []
        cs = lhz_core.create_constraintlist(Nl)
    elif isinstance(constraints[0][0], tuple):
        cs = constraints[:]
        constraints = []

    if len(constraints) == 0:
        qubit_dict = lhz_core.qubit_dict(Nl)
        for c in cs:
            constraints.append([qubit_dict[ci] for ci in c])

    assert isinstance(constraints[0][0], int)

    # prepare paulis for physical hamiltonians
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(Nfull):
        op_list = [si] * Nfull
        op_list[n] = sx
        sx_list.append(tensor(op_list))
        op_list[n] = sy
        sy_list.append(tensor(op_list))
        op_list[n] = sz
        sz_list.append(tensor(op_list))

    # hamiltonians
    HX = 0
    HY = 0
    HZ = 0
    for i in range(Np):
        HZ += Jijp[i] * sz_list[i]

    for i in range(Nfull):
        HX += sx_list[i]
        HY += sy_list[i]

    HCs = []
    NCs = []
    for c in constraints:
        h = 1
        for ci in c:
            h *= sz_list[ci]
        HCs.append(cstrength * (1 * 1 - h))
        NCs.append(1 - h)
        # h = -sz_list[c[0]]
        # for ci in range(len(c) - 1):
        #     h *= sz_list[c[ci + 1]]
        # HCs.append(cstrength * (1 + h))

    HC = sum(HCs)
    HP = HZ + HC

    Hdict = {'X': HX, 'Y': HY, 'Z': HZ, 'C': HC, 'P': HP,
             'x': sx_list, 'y': sy_list, 'z': sz_list, 'c': HCs,
             'Nc': sum(NCs)}
    return Hdict


def hdict_physical_not_full_sumconstraints(jij, constraints, sum_cos, cstrength=3.0):
    """
    Jij...             Jij matrix for physical spins,
                       format:  (check conventions)
                          - either linear, length N_phsical
                              eg. [J_01, J_02, J_03, J_12, J_13, J_23]
                          - or original Jij matrix (NlogicalxNlogical matrix)
    cstrength..        prefactor for all constraints
    constraints..      n-body constraints of the form 'cstrength*prod_i sigma_i^z'
                       constraint fulfilled, 0 energy,
                       constraint violated,  cstrength energy penalty
                       in total penalty = cstrength*num_violated_constraints
    :returns           H = sum{J_i sigma_i^z} + cstrength*sum{1-prod_i sigma_i^z}
    """
    # check and convert inputs
    n_p = len(jij)

    # prepare paulis for physical hamiltonians
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sm = sigmam()
    sp = sigmap()

    sx_list = []
    sy_list = []
    sz_list = []
    sp_list = []
    sm_list = []

    for n in range(n_p):
        op_list = [si] * n_p
        op_list[n] = sx
        sx_list.append(tensor(op_list))
        op_list[n] = sy
        sy_list.append(tensor(op_list))
        op_list[n] = sz
        sz_list.append(tensor(op_list))
        op_list[n] = sp
        sp_list.append(tensor(op_list))
        op_list[n] = sm
        sm_list.append(tensor(op_list))

    # hamiltonians
    HX = 0
    HZ = 0
    Htrans = 0

    involved_in_sum_constraints = list(set().union(*[set(co) for co in sum_cos]))

    for i in range(n_p):
        if i not in involved_in_sum_constraints:
            Htrans += sx_list[i]
        HX += sx_list[i]
        HZ += jij[i] * sz_list[i]

    Hexch = []
    for sum_co in sum_cos:
        for i in range(len(sum_co) - 1):
            temp = sm_list[sum_co[i]] * sp_list[sum_co[i + 1]]
        Hexch += [temp + temp.dag()]
        # print(sum_co[i], sum_co[i+1])
    # Hexch = Hexch + Hexch.dag()

    Nex = 0
    for i in involved_in_sum_constraints:
        Nex += sz_list[i]

    HCs = []
    NCs = []
    hhs = []
    for c in constraints:
        h = 1
        for ci in c:
            h *= sz_list[ci]
        hhs.append(h)
        HCs.append(cstrength * (1 - h) / 2)
        NCs.append((1 - h) / 2)

    HC = sum(HCs)
    HP = HZ + HC

    h_dict = {'X': HX, 'Z': HZ, 'C': HC, 'P': HP, 'test': hhs,
              'x': sx_list, 'y': sy_list, 'z': sz_list, 'c': HCs,
              'Nc': sum(NCs), 'T': Htrans, 'E': Hexch, 'N': Nex}
    return h_dict

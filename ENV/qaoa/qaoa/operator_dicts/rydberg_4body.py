from qutip import qeye, sigmax, sigmay, sigmaz, tensor, basis
import numpy as np
import lhz.core as lhz_core
from lhz.spinglass_utility import all_confs


def hdict_rydberg_4body(jij, constraints=None):
    """

    :param jij: Jij matrix for physical spins,
                       format:  (check conventions)
                          - either linear, length N_phsical
                              eg. [J_01, J_12, J_23, J_02, J_13, J_03]
                          - or original Jij matrix (NlogicalxNlogical matrix)
    :param constraints: n-body constraints of the form 'cstrength*prod_i sigma_i^z'
                        constraint fulfilled, 0 energy,
                        constraint violated,  cstrength energy penalty
                        in total penalty = cstrength*num_violated_constraints
    :return: H_P = sum{J_i sigma_i^z} + cstrength*sum{1-prod_i sigma_i^z}
    """
    # check and convert inputs
    shape = np.shape(jij)
    if len(shape) == 1:
        n_p = len(jij)
        jijp = jij

        assert n_p == lhz_core.n_physical(lhz_core.n_logical(n_p)), \
            'linear format of Jij wrong'

    elif len(shape) == 2:  # input as Jij matrix, test for list of lists
        n_p = lhz_core.n_physical(shape[0])
        jijp = np.array(jij)[np.triu_indices_from(jij, 1)]
    else:
        pass

    n_l = lhz_core.n_logical(n_p)

    # prepare constraints
    if constraints is None:
        constraints = []
        cs = lhz_core.create_constraintlist(n_l)
    elif isinstance(constraints[0][0], tuple):
        cs = constraints[:]
        constraints = []

    qubit_dict = lhz_core.qubit_dict(n_l)
    if len(constraints) == 0:
        for c in cs:
            constraints.append([qubit_dict[ci] for ci in c])

    assert isinstance(constraints[0][0], int)

    # prepare paulis for physical hamiltonians
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    d = basis(2, 1)
    pd = d * d.dag()
    u = basis(2, 0)
    pu = u * u.dag()

    sx_list = []
    sy_list = []
    sz_list = []
    pu_list = []
    pd_list = []

    for n in range(n_p):
        op_list = [si] * n_p
        op_list[n] = sx
        sx_list.append(tensor(op_list))
        op_list[n] = sy
        sy_list.append(tensor(op_list))
        op_list[n] = sz
        sz_list.append(tensor(op_list))
        op_list[n] = pu
        pu_list.append(tensor(op_list))
        op_list[n] = pd
        pd_list.append(tensor(op_list))

    # hamiltonians
    HX = 0
    HY = 0
    HZ = 0
    for i in range(n_p):
        HX += sx_list[i]
        HY += sy_list[i]
        HZ += jijp[i] * sz_list[i]

    HCs = []
    HC_fancy = []

    for c in constraints:
        h = 1
        for ci in c:
            h *= sz_list[ci]
        HCs.append(1.0 * (1 - h) / 2)

        # fancy rydberg part
        configs = all_confs(len(c))
        temp_co = [[], [], [], [], []]
        for conf in configs:
            # 0 in conf -> down, 1 in conf -> up
            conf_index = sum([(x + 1) // 2 for x in conf])
            conf_index = conf_index if len(c) == 4 else conf_index + 1

            proj = 1
            for i, ci in enumerate(c):
                if conf[i] == 1:
                    proj *= pu_list[ci]
                else:
                    proj *= pd_list[ci]

            temp_co[conf_index].append(proj.copy())

            # co_count = sum(conf)
            # co_check = co_count if len(c) == 4 else 1 + co_count
            # print(conf, conf_index, min(abs(co_check + 2), abs(co_check - 2)))
        HC_fancy.append(temp_co)
    # HC_fancy[i][j] at this point are projectors on states with j spins up for constraint plaquette i

    HC = sum(HCs)
    # HP = HZ + HC

    hdict = {'X': HX, 'Y': HY, 'Z': HZ, 'C': HC,
             'x': sx_list, 'y': sy_list, 'z': sz_list, 'c': HCs,
             'f': HC_fancy}
    return hdict


def hdict_rydberg_parity_gates(jij):
    """
    Translates given jij matrix into the odd parity scheme (does it) and creates projectors
    :param jij:
    :return:
    """
    hd_all = hdict_rydberg_4body(jij)

    # gathering projectors by number of spin up
    hc_temp = [0] * 5

    for co_site in hd_all['f']:
        for i, projs in enumerate(co_site):
            for proj in projs:
                hc_temp[i] += proj

    # odd subspace
    hd_all['O'] = hc_temp[1] + hc_temp[3]
    # F... four up or four down
    hd_all['F'] = hc_temp[0] + hc_temp[4]
    # T... two up/down
    hd_all['T'] = hc_temp[2]

    return hd_all


if __name__ == "__main__":
    nl = 4
    jiij = np.random.random((nl * (nl - 1) // 2,))
    hd = hdict_rydberg_4body(jiij)

    # check 1: reconstruct original constraint hamiltonian with projectors
    HC_proj_even = 0
    HC_proj_odd = 0
    for c_sites in hd['f']:
        for k, projs in enumerate(c_sites):
            for proj in projs:
                if k % 2 == 0:
                    HC_proj_even += proj
                else:
                    HC_proj_odd += proj
    print((nl - 2) * (nl - 1) // 2 - HC_proj_even == hd['C'])
    print(HC_proj_odd == hd['C'])

    # check 2:
    test = hdict_rydberg_parity_gates(jiij)
    print((nl - 2) * (nl - 1) // 2 - test['T'] - test['F'] == hd['C'])
    print(test['O'] == hd['C'])

    # %%
    # from lhz.spinglass_utility import create_qutip_wf, translate_configuration_logical_to_physical
    # from qutip import expect

    # #%%
    # state = [-1, -1, -1, -1, 1, 1]
    # print("state", state)
    # for blub in hd['o'][0]:
    #     print(expect(blub, create_qutip_wf(state)))
    #
    # #%%
    # for state in all_confs((nl * (nl - 1) // 2)):
    #     pass
    # #%%
    # for asdf in all_confs(nl):
    #     print('state: ', asdf)
    #     j = 1
    #     for blub in hd['o']:
    #         print("constraint", j)
    #         for i, x in enumerate(blub):
    #             if i in [1, 3]:
    #                 print('odd:  ', expect(x, create_qutip_wf(translate_configuration_logical_to_physical(asdf))))
    #             else:
    #                 print('even: ', expect(x, create_qutip_wf(translate_configuration_logical_to_physical(asdf))))
    #         j += 1
    #     print("-------")

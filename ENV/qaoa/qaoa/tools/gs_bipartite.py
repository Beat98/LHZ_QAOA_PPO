import numpy as np
from lhz.spinglass_utility import get_lowest_states


def gs_bipartite(local_fields, n_left=None, n_right=None, return_full_jij=False):
    """
    Returns ground-state and ground-state energy of a bipartite lattice
    :param return_full_jij:
    :param local_fields: local fields of bipartite lattice in standard LHZ format,
    eg numerated as:
    #      04  05  06  07  14  15  16  17  24  25  26  27  34  35  36  37
    #      02  03  04  05  06  07  08  09  12  13  14  15  16  17  18  19
    #      08  09  18  19  28  29  38  39  48  49  58  59  68  69  78  79
    :param n_left: If both n are None assuming square grid
    :param n_right:
    :return: gs_energy, gs_states
    """
    print('WARNING: seems to give wrong results!!!!!!!!!!!!!!')
    if n_left is None and n_right is None:
        n_left = int(np.sqrt(len(local_fields)))
        n_right = n_left
    n_l = n_left + n_right

    assert len(local_fields) == n_left * n_right

    jij = np.zeros((n_l, n_l))

    counter = 0
    for field in local_fields:
        i_l = counter // n_right
        i_r = n_left + (counter % n_right)

        jij[i_l, i_r] = field
        counter += 1

    jij_linear = jij[np.triu_indices_from(jij, 1)]

    if return_full_jij:
        return get_lowest_states(jij_linear), jij_linear
    else:
        return get_lowest_states(jij_linear)


def translate_conf(configuration, n_left, n_right):
    return [
        configuration[il]*configuration[ir+n_left] for il in range(n_left) for ir in range(n_right)
    ]


if __name__ == '__main__':
    # 0 1 2 3
    # 4 5 6 7
    # 8 9 0 1
    # 2 3 4 5

    loc = [+1, +1, +1, +1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1,  1]
    #      04  05  06  07  14  15  16  17  24  25  26  27  34  35  36  37
    #      02  03  04  05  06  07  08  09  12  13  14  15  16  17  18  19
    #      08  09  18  19  28  29  38  39  48  49  58  59  68  69  78  79

    gs0, states0 = gs_bipartite(loc)
    gs1, states1 = gs_bipartite(loc, 2, 8)
    gs2, states2 = gs_bipartite(loc, 8, 2)

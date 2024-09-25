from qaoa.QutipQaoa import QutipQaoa
import numpy as np
from lhz.core import n_physical
from lhz.qutip_hdicts import Hdict_physical
import multiprocessing as mp
from itertools import product
from time import time, sleep
import matplotlib.pyplot as plt
    

def do_qaoa(arg):
    return (
        arg[0],
        QutipQaoa(Hdict_physical(arg[1]), arg[2], arg[1], optimization_target="fidelity")
        .mc_optimization(n_maxiterations=None, temperature=0.01,
                                  return_timelines=True, rand_seed=arg[3])
    )


def do_qaoa2(arg):
    return (
        arg[0],
        QutipQaoa(Hdict_physical(arg[1]), arg[2], arg[1])
        .mc_optimization(n_maxiterations=2000, temperature=0.05,
                         return_timelines=True, rand_seed=arg[3])
    )


if __name__ == '__main__':
    Nl = 4
    Np = n_physical(Nl)
    runs = 3
    n_proc = 4

    p0 = 'ZXCX'
    pstrs = [p0*i for i in [1, 2, 3]]

    # testing params
    # pstrs = [p0, p0]
    # runs = 1

    Jl1 = 2*(np.random.random(size=(Np,)) - 0.5)
    Jl2 = 2*(np.random.random(size=(Np,)) - 0.5)
    # Jl1 = np.array([0.5, 1, 1, 0.5, 0.5, 1])
    # Jl2 = np.array([1, 1, 0.5, 0.5, 1, 0.5])
    Jls = [Jl1, Jl2]*runs

    parameters = list(product(Jls, pstrs))

    seeds = np.random.randint(low=0, high=666, size=(len(parameters,)))

    parameters = [(i, p[0], p[1], seed) for i, (p, seed) in enumerate(zip(parameters, seeds))]
    pstr_i_lookup = {}

    for parameter in parameters:
        pstr_i_lookup[parameter[0]] = np.where(parameter[2] == np.array(pstrs))[0][0]

    p = mp.Pool(processes=n_proc)

    tt = time()
    result = p.map_async(do_qaoa, parameters)
    # result = p.map_async(do_qaoa2, parameters)

    while not result.ready():
        sleep(1)

    if result.successful():
        print('calculation successful, took %.2fs' % (time()-tt))
        res = result.get()
    else:
        print('calculation unsuccessful, took %.2fs' % (time()-tt))


    # parse and plot results
    plt.subplots(len(pstrs), 1, True, figsize=(9, len(pstrs)*2))

    for i in range(len(pstrs)):
        plt.subplot(len(pstrs), 1, i+1)
        plt.ylim([0, 1])
        plt.title(pstrs[i])

    # plt.subplot(2, 1, 1)
    # plt.title('opt fid')
    # plt.plot(energy_vals2)
    # plt.ylabel('Energy')
    # plt.subplot(2, 1, 2)
    # plt.plot(fids2)
    # plt.ylim([0, 1])
    # plt.ylabel('|gs overlap|^2')
    # plt.xlabel('MC step')
    # plt.show()

    for re in res:
        plt.subplot(len(pstrs), 1, pstr_i_lookup[re[0]]+1)
        plt.plot(re[1].fidelities)
        # axes[pstr_i_lookup[re[0]]].plot(re[1][3])

    plt.show()






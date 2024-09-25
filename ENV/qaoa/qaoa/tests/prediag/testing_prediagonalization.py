from time import time
import lhz.qutip_hdicts as lq
from qaoa.QutipQaoa import QutipQaoa
from qutip import expect, sesolve, tensor, Qobj
import numpy as np
from qaoa.programs import qaoa_program
from lhz.spinglass_utility import create_qutip_wf, get_lowest_state, \
    translate_configuration_logical_to_physical, all_confs
import os
from lhz.core import n_logical
import scipy.sparse as sp

logical = True
logical = False

# definition of gate sequence:
if logical:
    progstr = 'PX'*6
else:
    progstr = 'ZCXZCX'
    progstr = 'ZXCX'*3
    # progstr = 'ZC'*6

# number of logical qubits
N = 5
Np = int(N * (N - 1) / 2)

t0 = time()

Jij = [0.1, -0.3, -0.7, -1.2, 0.8, -0.8, 1.2, -2.1, -2.1, 2.2, 1.0, 0.3, 0.7, 0.5, -0.6][:Np]
np.random.seed(234234)
Jij = 2*np.random.random(size=(Np,))-1
parameters = [-0.9, -0.1, -0.2, -0.8, 0.4, 0.2, 0.1, -1.1, 1.8, 0.3, 0.7, -0.6][:len(progstr)]
ee, ev = get_lowest_state(Jij)

if logical:
    hd = lq.Hdict_logical(Jij)
    gs = ev
    p0 = create_qutip_wf(['+'] * N)
else:
    hd = lq.Hdict_physical(Jij)
    gs = translate_configuration_logical_to_physical(ev)
    p0 = create_qutip_wf(['+'] * Np)

t1 = time()

# with sesolve
q = QutipQaoa(hd, progstr, Jij, psi0=p0)

t12 = time()

wf = q.run_qaoa(parameters)

t2 = time()

print('trun: %.2f, tprep: %.2f, val: %.3f' % (t2-t12, t12-t1, expect(hd['P'], wf)))

# preparing unitaries

fstrees = 'precalcees%d%s.npy' % (N, 'l' if logical else 'p')
fstrevs = 'precalcevs%d%s.npy' % (N, 'l' if logical else 'p')

if os.path.isfile(fstrees):  # load array
    print('loading...')
    ees = np.load(fstrees)
    evs = np.load(fstrevs)
else:  # diagonalize and save
    Hx = hd['X'].data.toarray()
    ees, evs = np.linalg.eigh(Hx)
    np.save(fstrees, ees)
    np.save(fstrevs, evs)

# ees, evs = np.linalg.eigh(Hx)  # why does it not work without hermitian?

# Hx = U.dag()*HxD*U

# U = Qobj(evs, dims=hd['X'].dims, shape=hd['X'].shape)
# Udag = U.dag()

# Us = U.data.toarray()
# Udags = Udag.data.toarray()
Us = evs
Udags = evs.conj().T
# HxD = np.exp(-1j*)
# print(U*U.dag())

wf2 = p0.copy()
wf3 = p0.copy()

t3 = time()
print('tprep = %.2f' % (t3 - t2))

tms = []
tss = []

for i in range(100):
    wf2s = p0.data.toarray()
    wf4 = p0.copy()

    tm = 0
    ts = 0

    for p, l in zip(parameters, progstr):
        # print(p, l)
        tt = time()
        if l == 'X':
            # wf2 = Udag*wf2
            # wf2 = Qobj(wf2.full()*np.exp(-1j*p*ees[:, np.newaxis]), dims=wf2.dims, shape=wf2.shape)
            # wf2 = U*wf2

            # wf2s = Us.dot(Udags.dot(wf2s).multiply(np.exp(-1j*p*ees[:, np.newaxis])))

            # wf2s = Us.dot(Udags.dot(wf2s)*(np.exp(-1j*p*ees[:, np.newaxis])))

            wf2s = Us @ (np.exp(-1j*p*ees[:, np.newaxis]) * (Udags @ wf2s))
        else:
            # wf2.data.data *= np.exp(-1j*p*hd[l].diag())
            wf2s = wf2s*np.exp(-1j*p*hd[l].diag()[:, np.newaxis])
        # wf2 = wf2 / np.sqrt(wf2.norm())
        tm += time()-tt  # print('murks', time()-tt)

        tt = time()
        res = sesolve(hd[l], wf4, [0, p], [])
        wf4 = res.states[-1]
        ts += time()-tt  # print('sesolve', time()-tt)

        # wf3 = (-1j*p*hd[l]).expm()*wf3

    # wf3 = (-1j*parameters[1]*hd['X']).expm()*(-1j*parameters[0]*hd['P']).expm()*p0

    # t4 = time()

    # print('1: %.3f, 2: %.3f, 3: %.3f' % (expect(hd['P'], wf), expect(hd['P'], wf2), expect(hd['P'], wf3)))
    tms += [tm]
    tss += [ts]
print('last run: ')
print('4: %.3f, t=%.2f, 2: %.3f, t=%.2f' % (expect(hd['P'], wf4), ts, expect(hd['P'], Qobj(wf2s, dims=p0.dims, shape=p0.shape)), tm))

tms = np.array(tms)
tss = np.array(tss)

print('summary:')
print('se: %.3f +- %.3f, mu: %.3f +- %.3f' % (tss.mean(), tss.std(), tms.mean(), tms.std()))
print('speedup: %.2f' % (tss.mean()/tms.mean()))
# for i in range(2**N):
#     print(evs[:, i].round(2).real)
#     print(hd['X'].eigenstates()[1][i].trans().full().round(2).real)
#     print('----')
#
# olap = np.zeros((2**N, 2**N), dtype=np.complex128)
#
# for i in range(2**N):
#     for j in range(2**N):
#         olap[i, j] = (evs[:, i]*hd['X'].eigenstates()[1][j])[0]
#
# print(olap.round(3).real)

from time import time
import lhz.qutip_hdicts as lq
from qutip import expect, sesolve, tensor, Qobj
import numpy as np
from qaoa.programs import qaoa_program
from lhz.spinglass_utility import create_qutip_wf, get_lowest_state, \
    translate_configuration_logical_to_physical, all_confs
from lhz.core import n_logical

logical = True

# definition of gate sequence:
if logical:
    progstr = 'PX'*6
else:
    progstr = 'ZCXZCX'
    progstr = 'ZXCX'*3

# number of logical qubits
N = 13
Np = int(N * (N - 1) / 2)

t0 = time()

Jij = [0.1, -0.3, -0.7, -1.2, 0.8, -0.8, 1.2, -2.1, -2.1, 2.2, 1.0, 0.3, 0.7, 0.5, -0.6][:Np]
np.random.seed(234234)
Jij = 2*np.random.random(size=(Np,))-1
parameters = [-0.9, -0.1, -0.2, -0.8, 0.4, 0.2, 0.1, -1.1, 1.8, 0.3, 0.7, -0.6][:len(progstr)]

if logical:
    hd = lq.Hdict_logical(Jij)
    p0 = create_qutip_wf(['+'] * N)
else:
    hd = lq.Hdict_physical(Jij)
    p0 = create_qutip_wf(['+'] * Np)

Hx = hd['X']
Hx = Hx.data.toarray()

ees, evs = np.linalg.eigh(Hx)  # why does it not work without hermitian?
U = Qobj(evs, dims=hd['X'].dims, shape=hd['X'].shape)
wf2 = p0.copy()

for p, l in zip(parameters, progstr):
    if l == 'X':
        wf2 = U.dag()*wf2
        # wf2.data.data = wf2.full()*np.exp(-1j*p*ees[:,np.newaxis])
        wf2 = Qobj(wf2.full()*np.exp(-1j*p*ees[:, np.newaxis]), dims=wf2.dims, shape=wf2.shape)
        wf2 = U*wf2
    else:
        wf2.data.data *= np.exp(-1j*p*hd[l].diag())
    wf2 = wf2 / np.sqrt(wf2.norm())

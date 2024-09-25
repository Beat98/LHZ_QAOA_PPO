from time import time
import lhz.qutip_hdicts as lq
from qaoa.QutipQaoa import QutipQaoa
import numpy as np
from lhz.spinglass_utility import create_qutip_wf

logical = True
logical = False

# definition of gate sequence:
if logical:
    progstr = 'PX'*10
else:
    progstr = 'ZCXZCX'
    progstr = 'ZXCX'*5
    progstr = 'ZXCYZXCYP'
    progstr = 'ZXUYVPX'
    progstr = 'ZZPCXCXYZVV'  # to check all cases for changes between bases

# number of logical qubits
N = 5
reps = 222
Np = int(N * (N - 1) / 2)

t0 = time()

np.random.seed(234234)
Jij = 2*np.random.random(size=(Np,))-1
# parameters = [-0.9, -0.1, -0.2, -0.8, 0.4, 0.2, 0.1, -1.1, 1.8, 0.3, 0.7, -0.6]*2
# parameters = parameters[:len(progstr)]
parameters = 2*np.random.random(size=(len(progstr),))-1

if logical:
    hd = lq.Hdict_logical(Jij)
    p0 = create_qutip_wf(['+'] * N)
else:
    hd = lq.Hdict_physical(Jij)
    p0 = create_qutip_wf(['+'] * Np)

tt = time()
qPre = QutipQaoa(hd, progstr, do_prediagonalization=True, jij=Jij)
print('diagonalization took %.2f s' % (time() - tt))

tt = time()
for i in range(reps):
    wfPre = qPre.run_qaoa_prediag(parameters)
tPre = time()-tt

tt = time()
for i in range(reps):
    wfQut = qPre.run_qaoa_qutip(parameters)
tQut = time()-tt

print('mean times:')
print('qutip: %.3fs, prediag: %.3fs' % (tQut/reps, tPre/reps))
print('total times:')
print('qutip: %.3fs, prediag: %.3fs' % (tQut, tPre))
print('speedup: %.2f' % (tQut/tPre))
print('overlap: %.3f + i %.3f ' % (wfPre.overlap(wfQut).real, wfPre.overlap(wfQut).imag))

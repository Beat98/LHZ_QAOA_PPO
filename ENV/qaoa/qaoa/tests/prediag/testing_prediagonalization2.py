from time import time
import lhz.qutip_hdicts as lq
import numpy as np
from lhz.spinglass_utility import create_qutip_wf, get_lowest_state, \
    translate_configuration_logical_to_physical, all_confs
import os
from lhz.core import n_logical
import scipy.sparse as sp

from qaoa.QutipQaoa import QutipQaoa

logical = True
# logical = False

# definition of gate sequence:
if logical:
    progstr = 'PX'*6
else:
    progstr = 'ZCXZCX'
    progstr = 'ZXCX'*3

# number of logical qubits
N = 10
Np = int(N * (N - 1) / 2)

t0 = time()

Jij = [0.1, -0.3, -0.7, -1.2, 0.8, -0.8, 1.2, -2.1, -2.1, 2.2, 1.0, 0.3, 0.7, 0.5, -0.6][:Np]
np.random.seed(234234)
Jij = 2*np.random.random(size=(Np,))-1
parameters = [-0.9, -0.1, -0.2, -0.8, 0.4, 0.2, 0.1, -1.1, 1.8, 0.3, 0.7, -0.6]*2
parameters = parameters[:len(progstr)]

if logical:
    hd = lq.Hdict_logical(Jij)
    p0 = create_qutip_wf(['+'] * N)
else:
    hd = lq.Hdict_physical(Jij)
    p0 = create_qutip_wf(['+'] * Np)

tt = time()
qPre = QutipQaoa(hd, progstr, do_prediagonalization=True)
print('diagonalization took %.2f s' % (time() - tt))

qQut = QutipQaoa(hd, progstr)

tt = time()
res1 = \
    qPre.mc_optimization(n_maxiterations=1000,
                         do_convergence_checks=False,
                         return_timelines=True,
                         measurement_functions=qPre.fidelity_function,
                         )

params1 = res1.parameters
final_energy1 = res1.final_objective_value
energy_vals1 = res1.objective_values
fids1 = res1.measurements

tPre = time()-tt

tt = time()
res2 = \
    qQut.mc_optimization(n_maxiterations=1000,
                         do_convergence_checks=False,
                         return_timelines=True,
                         measurement_functions=qQut.fidelity_function,
                         )

params2 = res2.parameters
final_energy2 = res2.final_objective_value
energy_vals2 = res2.objective_values
fids2 = res2.measurements

tQut = time()-tt

wfQ1 = qQut.run_qaoa(res1.parameters)
wfQ2 = qQut.run_qaoa(res2.parameters)
wfP1 = qPre.run_qaoa(res1.parameters)
wfP2 = qPre.run_qaoa(res2.parameters)

print(wfP1.overlap(wfQ1))
print(wfP2.overlap(wfQ2))


print('qutip: %.3f, prediag: %.3f' % (tQut, tPre))
print(final_energy1, final_energy2)

for i in range(len(progstr)):
    pass



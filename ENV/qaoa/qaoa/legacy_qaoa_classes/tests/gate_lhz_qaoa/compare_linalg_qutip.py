from qaoa.QutipQaoa import QutipQaoa
from qaoa.legacy_qaoa_classes.QutipGateLHZQaoa import QutipGateLHZQaoa

from lhz.qutip_hdicts import Hdict_physical
from lhz.core import standard_constraints

import numpy as np

nl = 5
# constraints = [sorted(c) for c in standard_constraints(nl)]
constraints = standard_constraints(nl)
n = nl*(nl-1)//2

Jij = [1, -0.56, 2, -1, 1, -1]*5
Jij = Jij[:n]

# aeshd = hd['P'].diag()
# aesl = hdl['P'].diag()
# aes = np.array(all_lhz_energies(Jij, constraints, 3))[::-1]

pstr = 'ZXCCXZ'
pstr = 'ZXC' * 2

hd = Hdict_physical(Jij)
# hdl = Hdict_logical(Jij)
qQ = QutipQaoa(hd, pstr, Jij)
qL = QutipGateLHZQaoa(pstr, Jij)

pars = ([0.1, 0.4, -1, 2, 3, 0.7]*2)[:len(pstr)]
# pars = [0.5, 0.2, 0.0]

qtg = ((-1j*hd['Z']*0.2).expm())
lag = qL._global_z(0.2)

qtC = ((-1j*hd['C']*0.2).expm())
laC = qL._full_constraints(0.2)
laCcorrectedphase = np.exp(1j*qL.cstr*0.2*len(constraints))*laC

diffC = qtC - laC
diffC2 = qtC - laCcorrectedphase

diff = qtg - lag
# print('diff global Z gate=', np.mean(abs(diff.data)))
# print('diff C gate=', np.mean(abs(diffC.data.toarray())), '+-', np.std(abs(diffC.data.toarray())))
# print('diff C2 gate=', np.mean(abs(diffC2.data.toarray())), '+-', np.std(abs(diffC2.data.toarray())))

sQ = qQ.execute_circuit(pars)
sL = qL.execute_circuit(pars)
# print(sQ)
# print(qtg*qQ.psi0)
# print(sL)
# print(abs(sQ.data.toarray()**2))
# print(abs(sL)**2)
print(qQ.objective_function(sQ))
print(qL.objective_function(sL))

from time import time
tt = time()
resQ = qQ.mc_optimization(n_maxiterations=3, return_timelines=True)
tQ = time()-tt
tt = time()
resL = qL.mc_optimization(n_maxiterations=3, return_timelines=True)
tL = time()-tt

print(tQ, tL)

from qaoa.result_plotter import plot_results
plot_results([resQ, resL], legend=['qutip', 'linalg'])
# plot_result(resQ, title='qutip')
# plot_result(resL, title='linalg')
print('\nqutip:\n', resQ)
print('\nlinalg:\n', resL)

pars = resL.parameters
sQ = qQ.execute_circuit(pars)
sL = qL.execute_circuit(pars)
print(qQ.objective_function(sQ))
print(qL.objective_function(sL))


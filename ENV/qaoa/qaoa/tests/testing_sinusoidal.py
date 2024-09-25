from time import time
import lhz.qutip_hdicts as lq
from qaoa.QutipQaoa import QutipQaoa
from qaoa.programs import get_startparameters
from qaoa.result_plotter import plot_result
import numpy as np
from lhz.spinglass_utility import create_qutip_wf, get_lowest_state
import matplotlib.pyplot as plt
from qutip import expect

logical = True
logical = False

# definition of gate sequence:
if logical:
    progstr = 'PX' * 5
else:
    progstr = 'ZCXZCX'
    progstr = 'ZXCX' * 5
    # progstr = 'ZCX' * 5
    # progstr = 'CX' * 7

# number of logical qubits
N = 5
Np = int(N * (N - 1) / 2)

t0 = time()

Jij = [0.1, -0.3, -0.7, -1.2, 0.8, -0.8, 1.2, -2.1, -2.1, 2.2, 1.0, 0.3, 0.7, 0.5, -0.6][:Np]
np.random.seed(234234)
Jij = 2 * np.random.random(size=(Np,)) - 1
# Jij = [1, 2, 3]

# some maxcut problem
Jij = np.array([[ 0,  0,  0,  0,  0],
                [-1./5,  0,  0,  0,  0],
                [-1, -1,  0,  0,  0],
                [ 0,  0, -1./3,  0,  0],
                [-1, -1,  0, -1,  0]], dtype=float)
Jij = -Jij - Jij.transpose()

Jij = 2 * np.random.random(size=(Np,)) - 1
# Jij = 0*Jij
# parameters = [-0.9, -0.1, -0.2, -0.8, 0.4, 0.2, 0.1, -1.1, 1.8, 0.3, 0.7, -0.6] * 2
# parameters = parameters[:len(progstr)]

if logical:
    hd = lq.Hdict_logical(Jij)
    # p0 = create_qutip_wf(['+'] * N)
else:
    hd = lq.Hdict_physical(Jij)
    # p0 = create_qutip_wf(['+'] * Np)

tt = time()
q = QutipQaoa(hd, progstr, do_prediagonalization=True, include_inverse_groundstate_in_fidelity=logical)
print('diagonalization took %.2f s' % (time() - tt))

# %%
tt = time()
eedi = int(str(tt).split('.')[1])
res1 = \
    q.mc_optimization(n_maxiterations=5000,
                      do_convergence_checks=True,
                      return_timelines=True,
                      rand_seed=eedi,
                      start_params=get_startparameters(progstr))

plot_result(res1)
print(res1)
pars = res1.parameters.copy()
print('gs: %.3f, qaoa: %.3f' % (q.gs_energy, res1.final_objective_value))

tPre = time() - tt


wf = q.run_qaoa(pars)
probs = np.round(abs(wf.data.toarray())**2, 2)
print('probs')
print(np.where(probs > 0.09))
# %%
stepspropi = 51
xxx = 15
steps = xxx * stepspropi
angles = np.linspace(-xxx * np.pi, xxx * np.pi, steps)

# pars = res1.parameters.copy()
which = [0, 1, 2, 3, 4, 5, 6, 7]

which = [0, 1, 2, 3]
# which = [i for i in range(len(progstr))]
# which = [1]

evals = np.zeros((steps, len(which)))

for ia, a in enumerate(angles):
    for iw, w in enumerate(which):
        ptemp = pars.copy()
        ptemp[w] += a
        # print(ptemp)
        evals[ia, iw] = expect(hd['P'], q.run_qaoa(ptemp))

plt.plot(angles/np.pi, evals)
plt.legend(['%d: %s' % (w, progstr[w]) for w in which])
plt.xlabel('delta angle')
plt.ylabel('energy')
plt.axhline(q.gs_energy, color='black', linestyle='dashed')
plt.axhline(res1.final_objective_value, color='grey', linestyle='dashed')
plt.grid(axis='x')
# for i in range(xxx):
#     plt.axvline(i * np.pi, color='grey', linestyle='dashed')
#     plt.axvline(-i * np.pi, color='grey', linestyle='dashed')
plt.show()

# # %%
# i = 2
#
#
# def test(p, wi):
#     # ptemp = pars.copy()
#     # ptemp[wi] = p
#     ptemp = p
#     return expect(hd['P'], q.run_qaoa(ptemp)[0])
#
#
# from scipydirect import minimize
#
# resdirect = minimize(test, bounds=[(-xxx * np.pi, xxx * np.pi)]*len(progstr), args=(i,), algmethod=1, disp=False, maxf=3000)
# print(resdirect)

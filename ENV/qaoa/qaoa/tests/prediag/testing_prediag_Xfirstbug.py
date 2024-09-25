from qaoa.QutipQaoa import QutipQaoa
from time import time
import numpy as np
from lhz.qutip_hdicts import Hdict_physical, Hdict_logical


# settings
nl = 4
nph = nl*(nl-1)//2

logical = False

jij = 2*np.random.random((nph, )) - 1

if logical:
    pstr = 'PX' * 3
    hd = Hdict_logical(jij)
else:
    pstr = 'ZCX' * 2
    pstr = 'UPX'
    pstr = 'PXUZ'
    # pstr = 'ZCX'
    hd = Hdict_physical(jij)

# pars = list(np.random.random((len(pstr), )))

compQs = []
compQs += [QutipQaoa(hd, pstr, jij, do_prediagonalization=True)]
compQs += [QutipQaoa(hd, 'X' + pstr, jij, do_prediagonalization=True)]
compQs += [QutipQaoa(hd, pstr, jij, do_prediagonalization=False)]
compQs += [QutipQaoa(hd, 'X' + pstr, jij, do_prediagonalization=False)]

mc_opti_opts = {'n_maxiterations': 2000, 'return_timelines': True, 'do_convergence_checks': False, 'par_change_range': 0.5}

times = []
rs = []
for q in compQs[:2]:
    t0 = time()
    rs += [q.mc_optimization(**mc_opti_opts)]
    times += [time() - t0]

# for t, r in zip(times, rs):
#     print(t, r)

shortparam = rs[0].parameters
longparam = rs[1].parameters

es = []
for q in compQs:
    if len(q.program.linearparameters) == len(pstr):  # short version
        p1 = shortparam
        p2 = longparam[1:]
    else:
        p1 = [0.0] + shortparam
        p2 = longparam

    es += [(q.energy_function(q.run_qaoa(p1)), q.energy_function(q.run_qaoa(p2)))]

for q, e in zip(compQs, es):
    print(q.program.program_string, e, q.run_qaoa)

# print('overlap: ', wPre.overlap(wNop))
# print('energies: 1: %.2f, 2: %.2f' % (qPre.energy_function(wPre), qNop.energy_function(wNop)))
# print('prep time, pred: %.3f, nop: %.3f' % (tPrediag, tnop))
# print('run time, pred: %.3f, nop: %.3f' % (tPreRun, tNopRun))

aaa = []
for angle in np.arange(0, 8, 0.1):
    p = longparam.copy()
    p[0] = angle
    aaa.append(compQs[1].energy_function(compQs[1].run_qaoa(p)))

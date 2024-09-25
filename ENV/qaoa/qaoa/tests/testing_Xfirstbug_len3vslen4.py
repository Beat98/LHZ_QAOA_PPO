# from qaoa.base_class import qaoa, Result
from qaoa.QutipQaoa import QutipQaoa
from qaoa.QaoaBase import Result
from time import time
import numpy as np
from lhz.qutip_hdicts import Hdict_physical, Hdict_logical
from qutip import expect
from copy import deepcopy

# settings
nl = 4
nph = nl*(nl-1)//2

logical = False

jij = 2*np.random.random((nph, )) - 1

if logical:
    pstr = 'PX' * 3
    hd = Hdict_logical(jij)
else:
    # pstr = 'UPX'
    hd = Hdict_physical(jij)


q1 = QutipQaoa(hd, 'UPX', jij, do_prediagonalization=True)
q2 = QutipQaoa(hd, 'XUPX', jij, do_prediagonalization=True)

# q1 = QutipQaoa(hd, 'UPX', jij, do_prediagonalization=False)
# q2 = QutipQaoa(hd, 'XUPX', jij, do_prediagonalization=False)


ress = []
for q in [q1, q2]:
    res = Result()
    res.final_objective_value = np.inf

    for it in range(10):
        r = q.mc_optimization(25*len(q.program.program_string)**2, return_timelines=False, temperature=0.02)

        if r.final_objective_value < res.final_objective_value:
            res = r
    ress += [deepcopy(res)]
    # print(res)

# now this is strange
print(q1.energy_function(q1.run_qaoa(ress[1].parameters[1:])))
print(q2.energy_function(q2.run_qaoa(ress[1].parameters)))
# print(expect(q2.Hdict['P'], q2.run_qaoa(ress[1].parameters)[0]))
# print(expect(q1.Hdict['P'], q1.run_qaoa(ress[1].parameters[1:])[0]))
